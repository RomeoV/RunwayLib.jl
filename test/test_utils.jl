module TestUtils

export retry_test

function retry_test(f, nretries)
    for i in 0:nretries
        try
            f()
            return
        catch e
            if i >= nretries
                rethrow(e)
            end
            @warn "Test failed, retrying ($i/$nretries)"
        end
    end
end

end  # module
