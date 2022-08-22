function rnd_num = number_gen(a, b, mode)
    if mode == 1
        % integers between [a,b]
        rnd_num = randi([a,b],1,1);
    else
        % float between [a,b]
        rnd_num = a + (b-a).*rand;
    end
end
