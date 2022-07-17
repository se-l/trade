module ClientTsHdf5

using JLD2
using Nettle
using JSON3

path_hdf5 = "C://repos//trade//data//ts2hdf5"

function get(meta)
    jldopen(path(meta), "r") do file
        return file["data"]
    end
end

fname(meta) = return hexdigest("md5", JSON3.write(meta)) * ".jld2"
path(meta) = return joinpath(path_hdf5, fname(meta))

function query(meta, params)
    """Given meta, get object. Given params return right time slice"""
end

function upsert(meta, data)
    fpath = path(meta)
    if ispath(fpath)
        d0 = missing
        jldopen(fpath, "r") do file
            d0 = file["data"]
        end
        ixd0_not_in_1 = findall(!(in(data[:, 1])), d0[:, 1])
        d1 = vcat(d0[ixd0_not_in_1, :], data)
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
# ClientTsHdf5.upsert(Dict("a"=>1), [[1,2,3] [2,3,4]])
# println(ClientTsHdf5.get(Dict("a"=>1)))
# ClientTsHdf5.delete(Dict("a"=>1))