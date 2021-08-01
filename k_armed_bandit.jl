using Gadfly
using Distributions: maximum
using Base: AbstractFloat, Float64, Integer, pause, Real
using Random, Distributions

function getdistr(μ::AbstractFloat ,σ::AbstractFloat; distr=Normal)
    return distr(μ, σ)
end


function getkarmeddict(k::Number, μ, σ;getdistr=getdistr, μnoise_μ=0.3, σnoise_μ=0.8, μnoise_σ=0.0, σnoise_σ=0.1)
    karmeddict = Dict()
    highest = 1
    highest_μ = -1*10e11

    for i in 1:k
        μ = μ + rand(getdistr(μnoise_μ,σnoise_μ))
        σ = σ + rand(getdistr(μnoise_σ,σnoise_σ))
        if highest_μ < μ
            highest_μ = μ
            highest = i
        end
        karmeddict[i] = getdistr(μ, σ)
    end
    println("Highest is $highest, with μ of $highest_μ")
    return karmeddict
end
# k_armed_dict = getkarmeddict(10, 0.0, 1.0)

function create_pre_state_t(k)
    return [[i for i in 1:k] [1 for i in 1:k] [[0.0] for i in 1:k]]
end
# pre_state_t = create_pre_state_t(10)

# array_Qₜa = zeros(10, 3)
# Now comes a part of building an actual bandits.
# I have to create a structure bandit, that will 
# hold predicted value - Qₜ(a), where a is [1:10]. So one value from karmeddict. 
# Also Qₜ(a) should be able to hold multiple functions.
# 
# To do that we need: 
"""
A structure Bandit that will hold:
 Qₜ(a) that will have a function that will calculate predicted value.
 Aₜ that will be maxargₐ. It will give a that maximizes the Qₜ(a).
A functions that will calculate Qₜ(a).
So I just need to build enough functions for each thing specificaly. 
Than I need to build structure that will hold the most important, expected value and 
"""

function getAₜ(array_Qₜa)
    Aₜ = zeros(1, size(array_Qₜa)[2])
    # Will recieve array with all the actions, so k predictions of actions. Than select the one with highest score.
    for i in 1:size(array_Qₜa)[2]
        aₘₐₓ = maximum(array_Qₜa[:, i])
        index = findall(x->x==aₘₐₓ,(array_Qₜa[:, i]))

        if length(index) > 1
            index = rand(index)
        end

        Aₜ[i] = Int(index[1])

    end
    return Aₜ
end


function t(a, pre_state_t)
    """Updates the array of all the times a was taken"""
    pre_state_t[a, 2] += 1
    return pre_state_t
end

#t(4, pre_state_t)

function qₜ(a, k_armed_dict, pre_state_t, save_array) 
    """
    Let's say that we get a=2, which gives than reward asociated with it.
    I found out that there is one little problem. It is that there are multiple different reward functions. 
    So the problem is that we need... Actually NO we don't. Because we can just create a function for each. 
    That would allow us to have the function for each specifically, but that isn't really time efficient writing for each specifically.
    BUT WE CAN MAKE IT SO THAT MAIN FUNCTION IS THE ONLY THING GETTING MODIFIED.
    """
    randnum = rand(k_armed_dict[a], 1)[1]
    pre_state_t = t(a, pre_state_t)
    push!(pre_state_t[a, 3], randnum)
    push!(save_array, randnum)
    return pre_state_t, save_array
end

""" In this part we need to get different Qₜ(a) functions. """

function Qₜa_simpleaverage(pre_state_t, array_Qₜa; line=1)
    """This will return a simple average of all the previous moves 
    and highest value. It pust in in array_Qₜa[:, 1]"""
    # return array_Qₜa[:, 1] + (cat(pre_state_t[:, 3], dims = size(pre_state_t)[1]) - array_Qₜa[:, 1])
    # This wouldn't work because all must be the same shape, else it will put them in some wierd collumns
    for i in 1:size(pre_state_t)[1]
        Qₖ₊₁_second = (pre_state_t[i, 3][end] - array_Qₜa[i, line])
        k⁻¹ = 1 / pre_state_t[i, 2] 
        array_Qₜa[i, line] = array_Qₜa[i, line] + Qₖ₊₁_second * k⁻¹
    end
    return array_Qₜa
end

function Qₜa_UTB(pre_state_t, array_Qₜa, c; Qₜa_simpleaverage=Qₜa_simpleaverage, line=2)
    # For this is required to have array_Qₜa[:, 1] done before
    for i in 1:size(pre_state_t)[1]
        t = sum(pre_state_t[:, 2])
        array_Qₜa[i, line] = Qₜa_simpleaverage(pre_state_t, array_Qₜa)[i, line] + c * sqrt(log(t)/pre_state_t[i, line])
    end
    return array_Qₜa
end

function Qₜa_OIV(array_Qₜa, optimistic_value, i; line=3)
    if i == 1
        array_Qₜa[:, 3] = fill(optimistic_value, size(array_Qₜa)[1])
    end

    return array_Qₜa
end

# In this step, we will test the code.

# building function to run it all

# array_Qₜa = zeros(10, 3)
# pre_state_t = create_pre_state_t(10)

# Qₜa_OIV(array_Qₜa, 5, pre_state_t, 1)

# array_Qₜa includes array(k, num_Qₜas). 
# In it we have current values of Qₜa.

function main(batch_size, k, μ, σ, c, optimistic_value; 
    create_pre_state_t=create_pre_state_t, getAₜ=getAₜ,
    getkarmeddict=getkarmeddict, t=t, qₜ=qₜ,
    Qₜa_simpleaverage=Qₜa_simpleaverage, Qₜa_UTB=Qₜa_UTB, Qₜa_OIV=Qₜa_OIV)
    
    """
    This function will put it all together, to calculate the output.
    Output will be the array of shape(batch_size, `number Qₜa-s`). 
    Now, here I will write the process for making this.
    1. We shall initialise all the variables that will be updated of stay constant.
    2. We will create a loop that will calculate our favoured step for each Qₜa. 
    3. Than repeat this batch_size times. Printing something every few steps.
    """
    array_Qₜa = zeros(k, 5)
    k_armed_dict = getkarmeddict(k, μ, σ)
    println("Build k_armed_dict")
    pre_state_t_1 = create_pre_state_t(k)
    pre_state_t_2 = create_pre_state_t(k)
    pre_state_t_3 = create_pre_state_t(k)
    println("Build pre_state_t-s")
    state1 = []
    state2 = []
    state3 = []
    println("Done initialising...")
    for i in 1:3
        sleep(0.4)
        println("Preparing...")
    end
    println("STARTING...")
    
    for i in 1:batch_size
        array_Qₜa = Qₜa_OIV(array_Qₜa, optimistic_value, i)
        array_Qₜa = Qₜa_simpleaverage(pre_state_t_3, array_Qₜa, line=3)
        array_Qₜa = Qₜa_simpleaverage(pre_state_t_1, array_Qₜa)
        array_Qₜa = Qₜa_UTB(pre_state_t_2, array_Qₜa, c)
        println("array_Qₜa prepared in round $i")

        Aₜ = getAₜ(array_Qₜa)

        pre_state_t_1, state1 = qₜ(Int(Aₜ[1]), k_armed_dict, pre_state_t_1, state1)
        pre_state_t_2, state2 = qₜ(Int(Aₜ[2]), k_armed_dict, pre_state_t_2, state2)
        pre_state_t_3, state3 = qₜ(Int(Aₜ[3]), k_armed_dict, pre_state_t_3, state3)

    end
    return state1, state2, state3

end

states = main(10000, 10, 0.0, 1.0, 2, 2)
states[1]
states[2]
states[3] 
size(states[1])[1]
plot(x=[i for i in 1:size(states[1])[1]], y=states[1], Geom.point, Geom.line) # The plot is not really great

