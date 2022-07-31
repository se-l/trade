__precompile__()

module ClientTsHdf5

using JLD2
using Nettle
using JSON3
using PyCall
using DataStructures
using Dates
np = pyimport("numpy")

path_hdf5 = "C://repos//trade//data//ts2hdf5"

dir_name(meta::AbstractDict) = return hexdigest("md5", JSON3.write(SortedDict(meta)))
f_path(dir_name::String, date::String) = return joinpath(path_hdf5, dir_name, date) * ".jld2"
f_path(meta::AbstractDict, date::String) = return joinpath(path_hdf5, dir_name(meta), date) * ".jld2"
dir_path(meta) = return joinpath(path_hdf5, dir_name(meta))
py"""
import numpy as np
def np_intersect1d(ar1, ar2, assume_unique=False, return_indices=False):
    # return zero based indexed arrays. Apply .+ 1
    return np.intersect1d(ar1, ar2, assume_unique=assume_unique, return_indices=return_indices)
"""

function query(meta::AbstractDict; start="", stop="9")
    """ToDo accept DateTimes and filter vcat for that"""
    dir = dir_path(meta)
    if !ispath(dir)
        return []
    else
        mats = []
        for fn in readdir(dir)
            date = replace(fn, ".jld2" => "")
            if start <= date <= stop
                push!(mats, load(joinpath(dir, fn), "data"))
            end
        end
        return vcat(mats...)
    end
end


function upsert(meta::AbstractDict, m_incoming; assume_sorted=true)
    map_date2Ix = DefaultDict{AbstractString, Vector{Int}}(() -> Vector{Int}())
    for (i, ts) in enumerate(m_incoming[:, 1])
        push!(map_date2Ix[string(Date(ts))], i)
    end
    for (date, indices) in pairs(map_date2Ix)
        file_path = f_path(meta, date)
        in_partition = m_incoming[indices, :]
        
        if ispath(file_path)
            d0 = load(file_path, "data")
            if assume_sorted & (d0[end, 1] < in_partition[1, 1])
                d1 = vcat(d0, in_partition)
            else
                _, ix1, ix2 = py"np_intersect1d"(d0[:, 1], in_partition[:, 1], assume_unique=true, return_indices=true)
                d1 = vcat(d0[setdiff(1:size(d0)[1], ix1 .+ 1), :], in_partition)
            end
            d1 = d1[sortperm(d1[:, 1]), :]
        else
            d1 = in_partition
            register(meta)
        end
        save(file_path, Dict("data" => d1))
    end
end

function delete(meta::AbstractDict; start="", stop="9")
    dir = dir_path(meta)
    for fn in readdir(dir)
        date = replace(fn, ".jld2" => "")
        if start <= date <= stop
            rm(joinpath(dir, date))
        end
    end
end

function drop(meta::AbstractDict)
    deregister(dir_name(meta))
    rm(dir_path(meta), recursive=true)
end

function drop(key::String)
    deregister(key)
    rm(joinpath(path_hdf5, key), recursive=true)
end

function register(meta::AbstractDict)
    p = joinpath(path_hdf5, "registry.json")
    if !ispath(p)
        map = Dict()
    else
        map = copy(JSON3.read(read(p, String)))
    end
    dir_key = dir_name(meta)
    if dir_key in keys(map)
        return
    else
        map[Symbol(dir_key)] = meta
        open(p, "w") do io
            JSON3.write(io, map)
        end
    end
end

function deregister(key:: String)
    p = joinpath(path_hdf5, "registry.json")
    map = copy(JSON3.read(read(p, String)))
    delete!(map, Symbol(key))
    open(p, "w") do io
        JSON3.write(io, map)
    end
end

end

# TEST
# ClientTsHdf5.register(Dict("a"=>1))    
# ClientTsHdf5.deregister(ClientTsHdf5.fname(Dict("a"=>1)))
# using Dates
# ClientTsHdf5.upsert(Dict("a"=>2), [[DateTime(2022, 1, i, 1, 1, i) for i in 1:3] [2,3,4]])
# upsert(Dict("a"=>2), [[DateTime(2022, 1, 1, 1, 1, i) for i in 3:5] [2,3,4]])
# query(Dict("a"=>2), start="2022-01-01")
# delete(Dict("a"=>2))
# println(ClientTsHdf5.get(Dict("a"=>1)))
# ClientTsHdf5.query(Dict("a"=>2))
# ClientTsHdf5.delete(Dict("a"=>2))
# @time println(ClientTsHdf5.fname(Dict("a"=>2)))
# for information in ["bid_buy_size_imbalance_ratio",
#     "bid_buy_count_imbalance_ratio",
#     "bid_buy_size_imbalance_net",
#     "bid_buy_count_imbalance_net"]
#     r = ClientTsHdf5.delete(Dict(
#         "measurement_name" => "order book",
#         "exchange" => "bitfinex",
#         "asset" => "ethusd",
#         "information" => information,
#         "unit" => "size_ewm_sum",
#         "delta_size_ratio" =>  0.5
#     ),)
# end
