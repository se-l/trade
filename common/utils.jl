ffill(v) = return v[accumulate(max, [i*!(ismissing(v[i]) | isnan(v[i])) for i in 1:length(v)], init=1)]
isna(v) = return map((i)->(ismissing(i) | isnan(i)), v)
function bfill(v)
    vr = reverse(v)
    return reverse(vr[accumulate(max, [i*!(ismissing(vr[i]) | isnan(vr[i])) for i in 1:length(vr)], init=1)])
end