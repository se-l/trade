using Dates
using PyCall
using DataFrames
using IterTools

import YAML

push!(LOAD_PATH, "C://repos//trade//layers//bars")
import OrderBook

include("C://repos//trade//connector//ts2hdf5//client.jl")
const Client = ClientTsHdf5

datetime = pyimport("datetime")
np = pyimport("numpy")

push!(pyimport("sys")."path", "C://repos//trade")
# build one class that imports all py classes I need
push!(pyimport("sys")."path", "C://repos//trade//common")
Paths = pyimport("paths").Paths
push!(pyimport("sys")."path", "C://repos//trade//common//modules")
# logger = pyimport("logger").logger
push!(pyimport("sys")."path", "C://repos//trade//layers")
BitfinexReader = pyimport("bitfinex_reader").BitfinexReader
push!(pyimport("sys")."path", "C://repos//trade//layers//bars")
# derive_events = pyimport("imbalance_order_book").derive_events


ffill(v) = return v[accumulate(max, [i*!(ismissing(v[i]) | isnan(v[i])) for i in 1:length(v)], init=1)]
isna(v) = return map((i)->(ismissing(i) | isnan(i)), v)

function ix_every_delta(arr, delta)  # no NANs here.
    cumsum_ = vcat([0], cumsum(arr[1:end-1] - arr[2:end]))
    ix_events = Vector()
    actual_deltas = Vector()
    ix_event_old = 1
    while true
        ix_event = 1
        for (ix, el) in enumerate(cumsum_[ix_event_old:end])
            if abs(el) >= delta
                ix_event = ix + ix_event_old
                break
            end
        end
        # ix_event = argmax(abs.(cumsum_) .>= delta)
        if ix_event == 1
            break
        else
            actual_delta = cumsum_[ix_event]
            push!(ix_events, ix_event)
            push!(actual_deltas, actual_delta)
            cumsum_ .-= actual_delta
            # cumsum_[1:ix_event] .= 0
            ix_event_old = ix_event
        end
    end
    return ix_events, actual_deltas
end

function invert_lt_zero_ratio!(v)
    ix_ask_greater = v .< 1  # why 1 not 0
    v[ix_ask_greater] .= 1 ./ v[ix_ask_greater]
    return v
end

function derive_events(dfo; level=30)
    df = filter!(r -> r.level <= level, dfo)
    df = combine(DataFrames.groupby(df, ["timestamp", "side", "level"]), :size=>last, :count=>last)
    ix_ts = unique(df.timestamp) |> sort
    map_side = Dict([(-1, 1), (1, 2)])

    book = zeros(Union{Missing, Float64}, length(ix_ts), 2, level, 2)
    book .= missing
    for side in [-1, 1]
        for level in 1:30
            if length(df[(df.side.==side) .& (df.level.==level), "size_last"]) > 0
                book[:, map_side[side], level, 1] = df[(df.side.==side) .& (df.level.==level), "size_last"]
                book[:, map_side[side], level, 2] = df[(df.side.==side) .& (df.level.==level), "count_last"]
            end
        end
    end
    return book
end


function main()
    @time begin
    settings = YAML.load_file(Paths.layer_settings)
    for exchange in keys(settings)
        println("$(exchange)")
        for (asset, params) in pairs(settings[exchange])
            if asset != "ethusd"
                continue
            end
            println("Loading order book for $(exchange) - $(asset)")
            params = get(params, "order book", Dict())
            delta_size_ratio = get(params, "delta_size_ratio", 0)
            start = Date(2022, 2, 7)
            end_ = Date(2022, 2, 7)
            for i = 1:1 + Dates.value(end_-start)
                dt = start + Day(i)
                println("Running $(dt)")
                df_py::PyObject = BitfinexReader.load_quotes(asset, datetime.datetime(2022, 2, 8), datetime.datetime(2022, 2, 8))
                df::DataFrame = DataFrame(
                    timestamp=Nanosecond.(df_py.get("timestamp").astype(np.int64)) + DateTime(1970),
                    price=df_py.get("price").values,
                    size=df_py.get("size").values,
                    count=df_py.get("count").values,
                    side=df_py.get("side").values,
                )
                df.timestamp .= df.timestamp[1]
                ob = OrderBook.Book(df, get(params, "level_from_price_pct", 0), 0.1)  # need to change or infer this 1.
                level = 30
                println("$(asset) - Summarizing $(level) order book levels")
                df = OrderBook.create_order_book!(ob)
                arr = derive_events(df)

                ix_valid = 1
                for (side, level) in product(1:2, 1:level)
                    println("Side: $(side) - Level: $(level)")
                    
                    arr[:, side, level, 1] = ffill(arr[:, side, level, 1])
                    arr[:, side, level, 2] = ffill(arr[:, side, level, 2])

                    ix_valid = max(argmin(isna(arr[:, side, level, 1])), ix_valid)
                    ix_valid = max(argmin(isna(arr[:, side, level, 2])), ix_valid)
                end

                # arr, v_ts = arr[ix_valid:end, :, :, :], (Nanosecond.(map(((i)->i.astype(np.int64)), df.timestamp)) + DateTime(1970))[ix_valid:end]
                arr, v_ts = arr[ix_valid:end, :, :, :], df.timestamp[ix_valid:end]
                arr[isnan.(arr) .| ismissing.(arr)] .= 0

                alpha = 2/(level + 1)
                weights = reshape(map((i)->(1-alpha)^i, 0:level-1), (1, 1, level, 1))

                arr = reshape(sum(arr .* weights, dims=3), (:, 2, 2))
                m_size, m_count = arr[:, :, 1], arr[:, :, 2]
                min_pos_cnt = minimum(m_count[m_count .> 0])
                m_count = ifelse.(m_count .== 0, min_pos_cnt, m_count)  # avoid divide by 0 error
                m_count = ifelse.(m_count .== 0, min_pos_cnt, m_count)  # avoid divide by 0 errorda
                min_pos_size = minimum(m_size[m_size .> 0])
                m_size = ifelse.(m_size .== 0, min_pos_size, m_size)  # avoid divide by 0 error

                count_ratio = m_count[:, 1] ./ abs.(m_count[:, 2])
                size_ratio = m_size[:, 1] ./ abs.(m_size[:, 2])  # bid|buy / ask|sell
                v_count_ratio = invert_lt_zero_ratio!(count_ratio)
                v_size_ratio = invert_lt_zero_ratio!(size_ratio)
                v_size_net = m_size[:, 1] + m_size[:, 2]
                v_count_net = m_count[:, 1] - m_count[:, 2]  # bid - ask
                
                println("Get Delta steps")
                @time ix_events, actual_deltas = ix_every_delta(v_size_ratio, delta_size_ratio)
                v_ts = v_ts[ix_events]
                @time ix_ts_unique_last = findlast.(isequal.(unique(v_ts)), [v_ts])
                println("Ingesting Size, Count Ratios and Net from total $(length(v_size_ratio)), 
                        reduced to # events $(length(v_ts)) -> # unique last ts: $(length(ix_ts_unique_last))")
                v_ts = v_ts[ix_ts_unique_last]
                # somehow need to ensure here only 1 value per timestamp
                for (information, v) in pairs(Dict(
                    "bid_buy_size_imbalance_ratio" => v_size_ratio,
                    "bid_buy_count_imbalance_ratio" => v_count_ratio,
                    "bid_buy_size_imbalance_net" => v_size_net,
                    "bid_buy_count_imbalance_net" => v_count_net,
                ))
                    Client.upsert(
                        Dict(
                            "measurement_name" => "order book",
                            "exchange" => exchange,
                            "asset" => asset,
                            "information" => information,
                            "unit" => "size_ewm_sum",
                            "delta_size_ratio" => delta_size_ratio
                        ),
                        hcat(v_ts, v[ix_events][ix_ts_unique_last])
                    )
                end
            end
        end
    end
    println("Done")
end
end

main()