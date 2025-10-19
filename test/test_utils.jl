function retry_test(f, nretries)
    for i in 0:nretries
        try
            f()
            return
        catch e
            if i >= n_retries
                rethrow(e)
            end
            @warn "Test failed, retrying ($i/$n_retries)"
        end
    end
end
