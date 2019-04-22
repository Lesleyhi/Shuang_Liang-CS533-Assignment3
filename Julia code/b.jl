
# using Pkg
# Pkg.add("Dates")
import Dates;

# default_sizes = Int8[31, 32, 96, 97, 127, 128, 129, 191, 192, 229, 255, 256, 257,
# 319, 320, 321, 417, 479, 480, 511, 512, 639, 640, 767, 768, 769]
default_sizes = [550,900,650,800,500,1000,950,700,400,600,750,850,450]
#default_sizes = [3000]
nsizes = length(default_sizes)
Mflops_s = zeros(nsizes)
nsamples = 20
data_s = "["

#println("hahah:",2.0f-10 * 100000000000.0 / 1.0)

for i_sample = 1:nsamples
    global data_s
    data_s = string(data_s,"[")
    for isize = 1:nsizes
        n = default_sizes[isize]
        A = rand(n,n)
        B = rand(n,n)
        C = Matrix{Float64}(undef,n,n)

        # Time a "sufficiently long" sequence of calls to reduce noise
        Gflops_s = -1.0
        seconds = -1.0
	seconds1 = -1.0
        timeout = 0.1
        n_iterations = 1
        while true 
            # Warm-up
            C = A*B

            # Benchmark n_iterations runs of square_dgemm
            t0 = Dates.value(Dates.now());
            for it = 1:n_iterations
                C = A*B
            end
            t1 = Dates.value(Dates.now());
            seconds = Float64(t1-t0)/1000.0;
	    seconds1 = seconds / n_iterations;

            # compute Gflop/s rate
            Gflops_s = 2.0f-9 * n_iterations * n * n * n / seconds;
            if seconds > timeout
                println("seconds:",seconds)
                break;
            end    
            n_iterations = n_iterations* 2
        end
        Mflops_s[isize] = Gflops_s * 1000;
#        data_s = string(data_s , Mflops_s[isize])
        data_s = string(data_s , seconds1)
        if isize < nsizes
            data_s = string(data_s , ",")
        else
            if i_sample < nsamples    
                data_s = string(data_s , "],")
            else    
                data_s = string(data_s ,  "]")
            end
        end

        # t0 = Dates.value(Dates.Time(Dates.now()))
        println("Size:",n,"  Mflop/s: ", Mflops_s[isize])
    end
end
data_s = string(data_s , "]")
println("data array:")
println(data_s)

