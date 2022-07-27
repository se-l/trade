__precompile__()

module ClientTsHdf5
# To Improve: Upsert - Assume sorted. Therefore, just append if last ts < first of new data. 
using JLD2
using Nettle
using JSON3
using PyCall
np = pyimport("numpy")

path_hdf5 = "C://repos//trade//data//ts2hdf5"

function get(meta)
    jldopen(path(meta), "r") do file
        return file["data"]
    end
end

fname(meta) = return hexdigest("md5", JSON3.write(meta)) * ".jld2"
path(meta) = return joinpath(path_hdf5, fname(meta))
py"""
import numpy as np
def np_intersect1d(ar1, ar2, assume_unique=False, return_indices=False):
    # return zero based indexed arrays. Apply .+ 1
    return np.intersect1d(ar1, ar2, assume_unique=assume_unique, return_indices=return_indices)
"""

function query(meta, from=missing, to=missing)
    """Given meta, get object. Given params return right time slice"""
    fpath = path(meta)
    if !ispath(fpath)
        return missing
    else
        jldopen(fpath, "r") do file
            return file["data"]
        end
    end
end

function upsert(meta, data)
    fpath = path(meta)
    if ispath(fpath)
        d0 = load(fpath, "data")
        _, ix1, ix2 = py"np_intersect1d"(d0[:, 1], data[:, 1], assume_unique=true, return_indices=true)
        d1 = vcat(d0[setdiff(1:size(d0)[1], ix1 .+ 1), :], data)
        d1 = d1[sortperm(d1[:, 1]), :]
    else
        d1 = data
        register(meta)        
    end
    jldopen(fpath, "w") do file
        file["data"] = d1
    end
end

function delete(meta)
    deregister(fname(meta))
    rm(path(meta))
end

function register(meta)
    p = joinpath(path_hdf5, "registry.json")
    if !ispath(p)
        map = Dict()
    else
        map = copy(JSON3.read(read(p, String)))
    end
    map[Symbol(fname(meta))] = meta
    open(p, "w") do io
        JSON3.write(io, map)
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
# ClientTsHdf5.upsert(Dict("a"=>2), [[Date(2022, 1, i) for i in 7:9] [2,3,4]])
# println(ClientTsHdf5.get(Dict("a"=>1)))
# ClientTsHdf5.query(Dict("a"=>2))
# ClientTsHdf5.delete(Dict("a"=>2))
# @time println(ClientTsHdf5.fname(Dict("a"=>2)))
r = ClientTsHdf5.query(Dict(
    "measurement_name"=> "order book",
    "delta_size_ratio"=> 0.5,
    "exchange"=> "bitfinex",
    "unit"=> "size_ewm_sum",
    "information"=> "bid_buy_size_imbalance_net",
    "asset"=> "ethusd"
  ))
