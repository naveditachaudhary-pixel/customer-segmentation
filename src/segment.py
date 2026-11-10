"""Customer segmentation on The Complete Journey (84.51°).

One row per household, built from a year of grocery transactions:
how recently, how often and how much they shop, what's in the basket,
and how much they lean on promotions and store brands.

Then: K-Means (k chosen by silhouette), Ward hierarchical clustering as a
cross-check, PCA for a 2-D view, and demographic profiles of each segment.

Run from the repo root:  python src/segment.py
"""
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
import pyreadr
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "results"
BASE = "https://raw.githubusercontent.com/bradleyboehmke/completejourney/master/data/"
FILES = ["transactions.rds", "products.rda", "demographics.rda"]
SEED = 42
K_RANGE = range(2, 9)

FEATURES = {
    "recency_weeks":   "weeks since last trip",
    "trips":           "shopping trips in the year",
    "spend":           "total spend ($)",
    "basket_value":    "average spend per trip ($)",
    "promo_share":     "share of spend bought on promotion",
    "private_share":   "share of spend on store brands",
    "produce_share":   "share of spend on produce",
    "fuel_share":      "share of spend on fuel",
    "drug_gm_share":   "share of spend on drug / general merchandise",
    "departments":     "number of departments shopped",
}
LOG = ["trips", "spend", "basket_value", "recency_weeks"]


def load():
    DATA.mkdir(exist_ok=True)
    for f in FILES:
        if not (DATA / f).exists():
            print(f"downloading {f} ...")
            urllib.request.urlretrieve(BASE + f, DATA / f)
    first = lambda f: next(iter(pyreadr.read_r(DATA / f).values()))
    return first("transactions.rds"), first("products.rda"), first("demographics.rda")


def household_features(tx, products):
    tx = tx.merge(products[["product_id", "department", "brand"]], on="product_id", how="left")
    tx["promo"] = (tx.retail_disc + tx.coupon_disc + tx.coupon_match_disc) > 0
    tx["private"] = tx.brand.str.lower().eq("private")
    v = tx.sales_value

    def dept_spend(name):
        return v.where(tx.department == name, 0)

    tx = tx.assign(promo_v=v.where(tx.promo, 0), private_v=v.where(tx.private, 0),
                   produce_v=dept_spend("PRODUCE"), fuel_v=dept_spend("FUEL"),
                   drug_v=dept_spend("DRUG GM"))
    g = tx.groupby("household_id")
    f = pd.DataFrame({
        "recency_weeks": tx.week.max() - g.week.max(),
        "trips": g.basket_id.nunique(),
        "spend": g.sales_value.sum(),
        "promo_v": g.promo_v.sum(), "private_v": g.private_v.sum(),
        "produce_v": g.produce_v.sum(), "fuel_v": g.fuel_v.sum(), "drug_v": g.drug_v.sum(),
        "departments": g.department.nunique(),
    })
    f = f[f.spend > 0]
    f["basket_value"] = f.spend / f.trips
    for new, col in [("promo_share", "promo_v"), ("private_share", "private_v"),
                     ("produce_share", "produce_v"), ("fuel_share", "fuel_v"),
                     ("drug_gm_share", "drug_v")]:
        f[new] = (f[col] / f.spend).clip(0, 1)
    return f[list(FEATURES)]


def prepare(f):
    X = f.copy()
    for c in LOG:
        X[c] = np.log1p(X[c])
    return StandardScaler().fit_transform(X)


def name_segments(f):
    """Give each K-Means cluster a stable, readable name from its medians."""
    med = f.groupby("kmeans")[["recency_weeks", "fuel_share", "basket_value"]].median()
    names = {}
    names[med.recency_weeks.idxmax()] = "Lapsed"
    rest = med.drop(index=list(names))
    names[rest.fuel_share.idxmax()] = "Fuel & store-brand regulars"
    rest = rest.drop(index=rest.fuel_share.idxmax())
    names[rest.basket_value.idxmax()] = "Big-basket loyalists"
    for k in rest.index.difference(list(names)):
        names[k] = "Occasional shoppers"
    return f.kmeans.map(names)


ORDER = ["Big-basket loyalists", "Fuel & store-brand regulars", "Occasional shoppers", "Lapsed"]


def main():
    tx, products, demo = load()
    f = household_features(tx, products)
    X = prepare(f)
    OUT.mkdir(exist_ok=True)

    # choose k: silhouette + inertia for k = 2..8
    rows = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(X)
        # stability: do different random starts give the same segments?
        runs = [KMeans(n_clusters=k, n_init=10, random_state=r).fit_predict(X) for r in range(8)]
        stability = np.mean([adjusted_rand_score(runs[0], r) for r in runs[1:]])
        rows.append({"k": k, "inertia": km.inertia_,
                     "silhouette": silhouette_score(X, km.labels_, random_state=SEED),
                     "stability_ari": stability})
    ksel = pd.DataFrame(rows)
    ksel.round(4).to_csv(OUT / "k_selection.csv", index=False)
    print(ksel.round(3).to_string(index=False))

    # Silhouette peaks at k=2, but that split is just light vs heavy shoppers.
    # k=4 is almost as stable across random starts and gives segments that
    # differ in *what* they buy, not only how much.  See README.
    K = 4
    km = KMeans(n_clusters=K, n_init=50, random_state=SEED).fit(X)
    f["kmeans"] = km.labels_
    f["segment"] = name_segments(f)

    # Ward hierarchical clustering with the same k, as a cross-check
    hac = fcluster(linkage(X, method="ward"), t=K, criterion="maxclust") - 1
    f["hac"] = hac
    ari = adjusted_rand_score(f.kmeans, f.hac)

    pca = PCA(n_components=2, random_state=SEED).fit(X)
    pcs = pca.transform(X)
    f["pc1"], f["pc2"] = pcs[:, 0], pcs[:, 1]

    f.round(4).to_csv(OUT / "household_segments.csv")
    pd.DataFrame(pca.components_.T, index=list(FEATURES), columns=["pc1", "pc2"]) \
        .round(3).to_csv(OUT / "pca_loadings.csv")

    prof = f.groupby("segment")[list(FEATURES)].median().reindex(ORDER)
    prof.insert(0, "households", f.groupby("segment").size())
    prof.insert(1, "share_of_households", prof.households / len(f))
    prof["share_of_spend"] = f.groupby("segment").spend.sum() / f.spend.sum()
    prof.round(3).to_csv(OUT / "segment_profiles.csv")

    # demographics exist for 801 of the households
    d = f[["segment"]].join(demo.set_index("household_id"), how="inner")
    prof["households_with_demographics"] = d.groupby("segment").size().reindex(ORDER).fillna(0).astype(int)
    demo_rows = []
    for col in ["age", "income", "household_size", "kids_count"]:
        ct = pd.crosstab(d.segment, d[col], normalize="index")
        for seg, r in ct.iterrows():
            for level, share in r.items():
                demo_rows.append({"segment": seg, "variable": col, "level": str(level),
                                  "share": round(share, 4)})
    pd.DataFrame(demo_rows).to_csv(OUT / "segment_demographics.csv", index=False)

    print(f"\nk = {K}  households = {len(f)}  with demographics = {len(d)}")
    print(f"K-Means vs Ward agreement (adjusted Rand) = {ari:.3f}")
    print(pd.crosstab(f.segment, f.hac, rownames=["kmeans"], colnames=["ward"]))
    print(f"PCA variance explained: PC1 {pca.explained_variance_ratio_[0]:.1%}, "
          f"PC2 {pca.explained_variance_ratio_[1]:.1%}")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(prof.round(2).T)


if __name__ == "__main__":
    main()
