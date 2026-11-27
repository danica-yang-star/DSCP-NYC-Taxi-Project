import pyarrow.dataset as ds

data = ds.dataset(
    "/workspace/yli2955/dscp_nyc_taxi_shared/v1/samples",
    format="parquet"
)

valid = ds.field("valid_for_primary_model") == True
positive = ds.field("tips") > 0
zero = ds.field("tips") == 0

n = data.count_rows(filter=valid, use_threads=False)
n_positive = data.count_rows(filter=valid & positive, use_threads=False)
n_zero = data.count_rows(filter=valid & zero, use_threads=False)

print("A group:", n)
print("Positive tips:", n_positive)
print("Zero tips:", n_zero)
print("Positive tip rate:", n_positive / n)
