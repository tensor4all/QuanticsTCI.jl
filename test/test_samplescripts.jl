using Test
using Glob

# Directory where the script files are located
script_dir = joinpath(@__DIR__, "samplescripts")

# Get all .jl files in the directory
script_files = glob("*.jl", script_dir)

# Function to execute a script using a new Julia process
function run_script_in_new_process(script_file)
    cmd = `julia --project=@. --startup-file=no $script_file`
    result = read(cmd, String)  # Run the command and capture the output
    return result
end

# Test each script file
@testset "External Script Tests" begin
    for script_file in script_files
        @testset "Testing $script_file" begin
            try
                run_script_in_new_process(script_file)
                @test true  # If no error occurs
            catch e
                @test false  # If an error occurs
                println("Error in $script_file: ", e)
            end
        end
    end
end
