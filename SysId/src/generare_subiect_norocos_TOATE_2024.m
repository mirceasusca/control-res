function [t,u,y,INFO] = generare_subiect_norocos_TOATE_2024(pwd,m_sub,n_sub)
% Base function

    t = [];
    u = [];
    y = [];
    INFO = struct(...
        'enunt','',...
        'mentiuni_suplimentare','');

    if ~strcmp(pwd,'test')
        error('Parola incorecta.')
    end

    switch mod(m_sub,2)
        case 0
            %% PARAMETRICE
            m_sub = round(m_sub/2);

            tip_subiect = 'ss';

            [t,u,y,INFO] = gen_sub_met_param_sistem_mecanic_doua_zerouri_trei_poli(m_sub,n_sub,tip_subiect);
%             [t,u,y,INFO] = gen_sub_met_param_sistem_mecanic_un_zero(m_sub,n_sub,tip_subiect);
            % [t,u,y,INFO] = gen_sub_met_param_sistem_doua_rezervoare(m_sub,n_sub,tip_subiect);
            % [t,u,y,INFO] = gen_sub_met_param_sistem_ordin_trei(m_sub,n_sub,tip_subiect);
    
        case 1
            %% *NE*PARAMETRICE
            m_sub = round((m_sub+1)/2);
            
            [t,u,y,INFO] = gen_sub_ord_I_tangentei_zgomot_impulsiv(m_sub,n_sub);
            % [t,u,y,INFO] = gen_sub_ord_I_met_reg_treapta_zgomot_impulsiv(m_sub,n_sub);
            % [t,u,y,INFO] = gen_sub_ord_I_met_reg_impuls_zgomot_impulsiv(m_sub,n_sub);
            % [t,u,y,INFO] = gen_sub_ord_II_osc_treapta_zgomot_impulsiv(m_sub,n_sub);
            % [t,u,y,INFO] = gen_sub_ord_II_osc_impuls_zgomot_impulsiv(m_sub,n_sub);
            % [t,u,y,INFO] = gen_sub_ord_sup_met_tangentei_zgomot_impulsiv(m_sub,n_sub);
            % [t,u,y,INFO] = gen_sub_ord_sup_met_cohen_coon_zgomot_impulsiv(m_sub,n_sub);
            % [t,u,y,INFO] = gen_sub_ord_sup_met_neparam_treapta_zgomot_impulsiv(m_sub,n_sub);
            
        otherwise
            error('Caz eronat, reverifica intrarile.')
    
    end

    [t,u,y] = force_column_dataset(t,u,y);

    assert(~isempty(t))
    assert(~isempty(u))
    assert(~isempty(y))
    assert(~isempty(INFO.enunt))
end

%% FUNCTII GENERALE
function out = spab(N,bi,bj,num_points,p)
%SPAB

    if nargin == 4
        p=1;
    end
    
    reg = ones(1,N);
    out_aux = repmat(reg,1,p);
    
    while length(out_aux) < num_points
        b_nou = xor(reg(bi),reg(bj));
        reg = [b_nou reg(1:N-1)];    
        out_aux = [out_aux b_nou*ones(1,p)];
    end
    
    out = out_aux(1:num_points);

end

%%
function [t,u,y] = force_column_dataset(t,u,y)
%FORCE_COLUMN_DATASET Ensure that the dataset vectors are columns.
    
    [d1u,d2u] = size(u);
    if d2u > d1u
        u = u';
    end
    [d1y,d2y] = size(y);
    if d2y > d1y
        y = y';
    end
    [d1t,d2t] = size(t);
    if d2t > d1t
        t = t';
    end

end

%%
function [u,y] = add_impulsive_noise(...
    u,y,m_sub,n_sub,Taq_multiplier,tol_impulse,rel_ampl_impulse)
%ADD_IMPULSIVE_NOISE
    
    p2p_u = max(u)-min(u);
    p2p_y = max(y)-min(y);
    rng(2*m_sub+10*n_sub)
    for k = 1:length(u)
        r = Taq_multiplier*rand(1);
        if r <= Taq_multiplier*tol_impulse
            u(k) = u(k) + randn(1)*p2p_u*rel_ampl_impulse;
        end
        r = Taq_multiplier*rand(1);
        if r <= Taq_multiplier*tol_impulse
            y(k) = y(k) + randn(1)*p2p_y*rel_ampl_impulse;
        end
    end

end

%%
function [t,u,y] = gen_dataset_fcn_neparametrice(H,m_sub,n_sub,...
    u0,ust,N,Taq_multiplier,tol_extra_ts,tol_cut,reverse,...
    quantizer_relerr_bits,with_noise,noise_rel_tol,...
    with_imp_noise,tol_impulse,rel_ampl_impulse,...
    step_type,impulse_duty_cycle)
%GEN_STEP_DATASET_FCN -- Functie de baza pentru subiectele cu metode
%*ne*parametrice

    assert(N >= 2)
    assert(tol_extra_ts >= 0.8)  % recommended above 1-1.25
    assert(Taq_multiplier >= 50)
    
    perf = stepinfo(H);
    ts = perf.SettlingTime;
    assert(ts > 0, 'ts must be positive, system is problematic');
    Taq = ts/Taq_multiplier;
    
    % gen step signals
    N_aux = N+2; % one extra at the start, one at the end, which will be cut later
    
    num_samples_per_ts = round(ts*tol_extra_ts/Taq);
    
    u = [];
    
    if step_type
        % step signal
        for k=1:N_aux
            if mod(k,2) == 1
                u = [u,u0*ones(1,num_samples_per_ts)];
            else
                u = [u,ust*ones(1,num_samples_per_ts)];
            end
        end
    else
        % impulse signal
        assert(reverse == false,'reverse flag must be false for impulse input.')
        for k=1:N_aux
            u_partial = u0*ones(1,num_samples_per_ts);
            u_partial(1:round(num_samples_per_ts*impulse_duty_cycle)) = ust;
            u = [u,u_partial];
        end
    end
    
    % reverse u0 and ust, if imposed
    if reverse
        u_new = u;
        for k=1:length(u)
            if u(k) == u0
                u_new(k) = ust;
            else
                u_new(k) = u0;
            end
        end
        u = u_new;
    end
    
    % simulate with nonzero initial conditions
    y0 = evalfr(H,0)*u0;
    [num,den] = tfdata(H,'v');
    [A,B,C,D] = tf2ss(num,den);
    sys_ci = ss(A',C',B',D); % FCO, to impose x1(0) = y(0)
    t = (0:(length(u)-1))*Taq;
    y = lsim(sys_ci,u,t,[y0,zeros(1,length(pole(H))-1)]);
    
    % cut from N_aux signal according to tol_cut
    Num_samples_to_cut = num_samples_per_ts-round(num_samples_per_ts*tol_cut);
    %
    u = u(1:end-Num_samples_to_cut); % end
    u = u(Num_samples_to_cut:end); % beginning
    %
    y = y(1:end-Num_samples_to_cut); % end
    y = y(Num_samples_to_cut:end); % beginning
    
    t = (0:(length(u)-1))*Taq;
    
    % force all vectors to be columns
    [t,u,y] = force_column_dataset(t,u,y);
    
    if with_noise
        p2p_u = max(u)-min(u);
        u = u + (rand([length(u),1])-0.5)*p2p_u*noise_rel_tol;
        p2p_y = max(y)-min(y);
        y = y + (rand([length(y),1])-0.5)*p2p_y*noise_rel_tol;
    end
    
    % add impulsive noise, if applicable
    if with_imp_noise
        [u,y] = add_impulsive_noise(...
            u,y,m_sub,n_sub,Taq_multiplier,tol_impulse,rel_ampl_impulse);
    end
    
    % quantize signals
    [u,y] = quantize_signals(u,y,quantizer_relerr_bits,m_sub,n_sub);

end

%%
function [u,y] = quantize_signals(u,y,quantizer_relerr_bits,m_sub,n_sub)
%QUANTIZE_SIGNALS -- Used in gen_dataset_fcn_parametrice
    
    p2p_u = max(u)-min(u);
    adcu_step = p2p_u/(pow2(quantizer_relerr_bits)-1);
    rng(m_sub)
    u = round(u/adcu_step)*adcu_step;
    
    p2p_y = max(y)-min(y);
    adcy_step = p2p_y/(pow2(quantizer_relerr_bits)-1);
    rng(n_sub)
    y = round(y/adcy_step)*adcy_step;

end

%%
function [t,u,y] = gen_dataset_fcn_parametrice(H,m_sub,n_sub,...
    u0,ust,N,bi,bj,p,Taq_multiplier,tol_extra_ts,tol_cut,...
    quantizer_relerr_bits,with_noise,noise_rel_tol,...
    with_input_noise,num_steps,factor_decimare)
%GEN_STEP_DATASET_FCN -- Functie de baza pentru subiectele cu metode
%parametrice
    
    assert(num_steps>=2)
    assert(tol_extra_ts >= 0.8)  % recommended above 1-1.25
    assert(Taq_multiplier >= 50)
    
    perf = stepinfo(H);
    ts = perf.SettlingTime;
    if ~isfinite(ts)
        ts = perf.TransientTime;
    end
    if ~isfinite(ts)
        assert(ts > 0, 'ts must be positive, system is problematic');
    end
    Taq = ts/Taq_multiplier;
    
    % gen step signals
    N_aux = num_steps+2; % one extra at the start, one at the end, which will be cut later
    
    num_samples_per_ts = round(ts*tol_extra_ts/Taq)*N;
    
    u = [];

    % step signal
    for k=1:N_aux
        if mod(k,2) == 1
            u = [u,u0*ones(1,num_samples_per_ts)];
        else
            u = [u,ust*ones(1,num_samples_per_ts)];
        end
    end

    u_spab = spab(N,bi,bj,length(u),p);

    u = u + u_spab*(ust-u0)/5;
    
    % simulate with nonzero initial conditions
    y0 = evalfr(H,0)*u0;
    [num,den] = tfdata(H,'v');
    [A,B,C,D] = tf2ss(num,den);
    sys_ci = ss(A',C',B',D); % FCO, to impose x1(0) = y(0)
    t = (0:(length(u)-1))*Taq;
    y = lsim(sys_ci,u,t,[y0,zeros(1,length(pole(H))-1)]);
    
    % cut from N_aux signal according to tol_cut
    Num_samples_to_cut = num_samples_per_ts-round(num_samples_per_ts*tol_cut);
    %
    u = u(1:end-Num_samples_to_cut); % end
    u = u(Num_samples_to_cut:end); % beginning
    %
    y = y(1:end-Num_samples_to_cut); % end
    y = y(Num_samples_to_cut:end); % beginning
    
    t = (0:(length(u)-1))*Taq;
    
    % force all vectors to be columns
    [t,u,y] = force_column_dataset(t,u,y);
    
    if with_noise
        if with_input_noise
            p2p_u = max(u)-min(u);
            u = u + (rand([length(u),1])-0.5)*p2p_u*noise_rel_tol;
        end
        p2p_y = max(y)-min(y);
        y = y + (rand([length(y),1])-0.5)*p2p_y*noise_rel_tol;
    end

    % quantize signals
    [u,y] = quantize_signals(u,y,quantizer_relerr_bits,m_sub,n_sub);
    
    assert(factor_decimare>=1)
    t_ext = (0:(length(u)*factor_decimare-1))*(Taq/factor_decimare);
    u_ext = [];
    y_ext = [];
    for idx = 1:length(u)
        u_ext = [u_ext u(idx)*ones(1,factor_decimare)];
        y_ext = [y_ext y(idx)*ones(1,factor_decimare)];
    end
    
    % force all vectors to be columns
    [t,u,y] = force_column_dataset(t_ext,u_ext,y_ext);

end

%% SUBIECTE METODE *NE*PARAMETRICE
function [t,u,y,INFO] = gen_sub_ord_I_tangentei_zgomot_impulsiv(m_sub,n_sub)
% Metoda Tangentei, Ordin I, Zgomot impulsiv
% VALIDAT
    
    rng(m_sub+10*n_sub)
    K = 0.1+rand(1)*5;
    rng(10*m_sub+n_sub)
    T = 1e-4+9e-4*rand(1);
    H = tf(K,[T,1]);
    
    u0 = -2+3*rand(1);
    ust = u0+10*rand(1);
    N = 3;
    %
    Taq_multiplier = round(500+300*rand(1));
    tol_extra_ts = 1.2+0.5*rand(1);
    tol_cut = 0.08+0.05*rand(1);
    quantizer_relerr_bits = 6;
    %
    with_noise = true;
    noise_rel_tol = 0.03;
    %
    with_imp_noise = true;
    tol_impulse = 0.01/3/2;
    rel_ampl_impulse = 1/4;
    %
    step_type = true;
    reverse = false;
    impulse_duty_cycle = 0.04;
    
    [t,u,y] = gen_dataset_fcn_neparametrice(...
        H,m_sub,n_sub,u0,ust,N,...
        Taq_multiplier,tol_extra_ts,tol_cut,...
        reverse,quantizer_relerr_bits,with_noise,noise_rel_tol,with_imp_noise,...
        tol_impulse,rel_ampl_impulse,step_type,impulse_duty_cycle);
    
    INFO = struct(...
        'enunt',['Sa se identifice modelul matematic al circuitului descris ' ...
        'prin comportamentul intrare/iesire u(t)/y(t) utilizand metoda tangentei. ' ...
        'Semnalele sunt achizitionate cu un osciloscop, iar circuitul nu are ecranare corespunzatoare.'],...
        'mentiuni_suplimentare','Eroarea de identificare sa fie sub 5%.'...
        );
end

%%
function [t,u,y,INFO] = gen_sub_ord_I_met_reg_treapta_zgomot_impulsiv(m_sub,n_sub)
% Metoda Regresiei Liniare, Ordin I, Intrare Treapta
% VALIDAT
    
    rng(2*m_sub+20*n_sub)
    K = 0.1+rand(1)*5;
    rng(10*m_sub+n_sub)
    T = 1e-4+9e-4*rand(1);
    H = tf(K,[T,1]);
    
    u0 = -2+3*rand(1);
    ust = u0+10*rand(1);
    N = 3;
    %
    Taq_multiplier = round(500+300*rand(1));
    tol_extra_ts = 1.2+0.3*rand(1);
    tol_cut = 0.08+0.05*rand(1);
    quantizer_relerr_bits = 6;
    %
    with_noise = true;
    noise_rel_tol = 0.03;
    %
    with_imp_noise = true;
    tol_impulse = 0.01/3/2;
    rel_ampl_impulse = 1/10;
    %
    step_type = true;
    reverse = false;
    impulse_duty_cycle = 0.04;
    
    [t,u,y] = gen_dataset_fcn_neparametrice(...
        H,m_sub,n_sub,u0,ust,N,...
        Taq_multiplier,tol_extra_ts,tol_cut,...
        reverse,quantizer_relerr_bits,with_noise,noise_rel_tol,with_imp_noise,...
        tol_impulse,rel_ampl_impulse,step_type,impulse_duty_cycle);
    
    INFO = struct(...
        'enunt',['Sa se identifice modelul matematic al circuitului descris ' ...
        'prin comportamentul intrare/iesire u(t)/y(t) utilizand metoda regresiei liniare. ' ...
        'Semnalele sunt achizitionate cu un osciloscop.'],...
        'mentiuni_suplimentare','Eroarea de identificare sa fie sub 5%.'...
        );
end

%%
function [t,u,y,INFO] = gen_sub_ord_I_met_reg_impuls_zgomot_impulsiv(m_sub,n_sub)
% Metoda Regresiei Liniare, Ordin I, Intrare Impuls
% VALIDAT
    
    rng(2*m_sub+20*n_sub)
    K = 0.1+rand(1)*5;
    rng(10*m_sub+n_sub)
    T = 1e-4+9e-4*rand(1);
    H = tf(K,[T,1]);
    
    u0 = -2+3*rand(1);
    ust = u0+10*rand(1);
    N = 3;
    %
    Taq_multiplier = round(400+300*rand(1));
    tol_extra_ts = 1.2+0.5*rand(1);
    tol_cut = 0.08+0.05*rand(1);
    quantizer_relerr_bits = 6;
    %
    with_noise = true;
    noise_rel_tol = 0.03;
    %
    with_imp_noise = true;
    tol_impulse = 0.01/3/2;
    rel_ampl_impulse = 1/10;
    %
    step_type = false;
    reverse = false;
    impulse_duty_cycle = 0.02+rand(1)*0.08;
    
    [t,u,y] = gen_dataset_fcn_neparametrice(...
        H,m_sub,n_sub,u0,ust,N,...
        Taq_multiplier,tol_extra_ts,tol_cut,...
        reverse,quantizer_relerr_bits,with_noise,noise_rel_tol,with_imp_noise,...
        tol_impulse,rel_ampl_impulse,step_type,impulse_duty_cycle);
    
    INFO = struct(...
        'enunt',['Sa se identifice modelul matematic al circuitului descris ' ...
        'prin comportamentul intrare/iesire u(t)/y(t) utilizand metoda regresiei liniare. ' ...
        'Semnalele sunt achizitionate cu un osciloscop.'],...
        'mentiuni_suplimentare','Eroarea de identificare sa fie sub 5%.'...
        );
end

%%
function [t,u,y,INFO] = gen_sub_ord_II_osc_treapta_zgomot_impulsiv(m_sub,n_sub)
% Circuit ordin II, Intrare Treapta
% VALIDAT
    
    rng(m_sub+10*n_sub)
    K = 0.2+rand(1)*2;
    zeta = 0.15+0.35*rand(1);
    wn = 800+1000*rand(1);
    H = tf(K*wn^2,[1,2*zeta*wn,wn^2]);
    
    u0 = -3+4*rand(1);
    ust = u0+0.2+5*rand(1);
    N = 2;
    %
    Taq_multiplier = round(400+500*rand(1));
    tol_extra_ts = 1.3+0.6*rand(1);
    tol_cut = 0.08+0.06*rand(1);
    quantizer_relerr_bits = 6;
    %
    with_noise = true;
    noise_rel_tol = 0.03;
    %
    with_imp_noise = true;
    tol_impulse = 0.01/3/4;
    rel_ampl_impulse = 1/4;
    %
    step_type = true;
    if mod(n_sub,2)==0
        reverse = false;
    else
        reverse = true;
    end
    impulse_duty_cycle = 0.02+rand(1)*0.08;
    
    [t,u,y] = gen_dataset_fcn_neparametrice(...
        H,m_sub,n_sub,u0,ust,N,...
        Taq_multiplier,tol_extra_ts,tol_cut,...
        reverse,quantizer_relerr_bits,with_noise,noise_rel_tol,with_imp_noise,...
        tol_impulse,rel_ampl_impulse,step_type,impulse_duty_cycle);
    
    INFO = struct(...
        'enunt',['Sa se identifice modelul matematic al circuitului descris ' ...
        'prin comportamentul intrare/iesire u(t)/y(t) utilizand o metoda neparametrica. ' ...
        'Semnalele sunt achizitionate cu un osciloscop, iar circuitul nu are ecranare corespunzatoare.'],...
        'mentiuni_suplimentare','Eroarea de identificare sa fie sub 10%.'...
        );
end

%%
function [t,u,y,INFO] = gen_sub_ord_II_osc_impuls_zgomot_impulsiv(m_sub,n_sub)
% Circuit ordin II, Intrare Impuls
% VALIDAT
    
    rng(m_sub+10*n_sub)
    K = 0.2+rand(1)*2;
    zeta = 0.15+0.35*rand(1);
    wn = 800+1000*rand(1);
    H = tf(K*wn^2,[1,2*zeta*wn,wn^2]);
    
    u0 = -3+4*rand(1);
    ust = u0+1+5*rand(1);
    N = 2;
    %
    Taq_multiplier = round(400+500*rand(1));
    tol_extra_ts = 1.3+0.6*rand(1);
    tol_cut = 0.08+0.06*rand(1);
    quantizer_relerr_bits = 6;
    %
    with_noise = true;
    noise_rel_tol = 0.03;
    %
    with_imp_noise = true;
    tol_impulse = 0.01/3/4;
    rel_ampl_impulse = 1/4;
    %
    step_type = false;
    reverse = false;
    impulse_duty_cycle = 0.04+rand(1)*0.06;
    
    [t,u,y] = gen_dataset_fcn_neparametrice(...
        H,m_sub,n_sub,u0,ust,N,...
        Taq_multiplier,tol_extra_ts,tol_cut,...
        reverse,quantizer_relerr_bits,with_noise,noise_rel_tol,with_imp_noise,...
        tol_impulse,rel_ampl_impulse,step_type,impulse_duty_cycle);
    
    INFO = struct(...
        'enunt',['Sa se identifice modelul matematic al circuitului descris ' ...
        'prin comportamentul intrare/iesire u(t)/y(t) utilizand o metoda neparametrica. ' ...
        'Semnalele sunt achizitionate cu un osciloscop, iar circuitul nu are ecranare corespunzatoare.'],...
        'mentiuni_suplimentare','Eroarea de identificare sa fie sub 10%.'...
        );
end

%%
function [t,u,y,INFO] = gen_sub_ord_sup_met_tangentei_zgomot_impulsiv(m_sub,n_sub)
% Proces industrial (ordin superior), metoda tangentei
% VALIDAT
    
    rng(m_sub+10*n_sub)
    K = 2.2+rand(1)*2;
    T1 = 20+50*rand(1);
    H = tf(K,[T1,1]);

    T_aux = T1*(1+1.8*(rand(1)-0.5)); % intre 20% si 180% din T1.
    H = series(H,tf(1,[T_aux,1]));

    rng(m_sub+10*n_sub)
    r = rand(1);  % posibil ordin 3
    if r < 0.5
        T_aux = T1*(1+1.9*(rand(1)-0.5));
        H = series(H,tf(1,[T_aux,1]));
    end

    r = rand(1);  % posibil ordin 4
    if r < 0.5
        T_aux = T1*(1+1.9*(rand(1)-0.5));
        H = series(H,tf(1,[T_aux,1]));
    end

    r = rand(1);  % posibil ordin 5
    if r < 0.5
        T_aux = T1*(1+1.9*(rand(1)-0.5));
        H = series(H,tf(1,[T_aux,1]));
    end

    u0 = -3+4*rand(1);
    ust = u0+1+5*rand(1);
    N = 3;
    %
    Taq_multiplier = round(400+500*rand(1));
    tol_extra_ts = 1.3+0.6*rand(1);
    tol_cut = 0.08+0.06*rand(1);
    quantizer_relerr_bits = 6;
    %
    with_noise = true;
    noise_rel_tol = 0.03;
    %
    with_imp_noise = true;
    tol_impulse = 0.01/3/4;
    rel_ampl_impulse = 1/4;
    %
    step_type = true;
    if mod(n_sub,2)==0
        reverse = false;
    else
        reverse = true;
    end
    impulse_duty_cycle = 0.04+rand(1)*0.06;
    
    [t,u,y] = gen_dataset_fcn_neparametrice(...
        H,m_sub,n_sub,u0,ust,N,...
        Taq_multiplier,tol_extra_ts,tol_cut,...
        reverse,quantizer_relerr_bits,with_noise,noise_rel_tol,with_imp_noise,...
        tol_impulse,rel_ampl_impulse,step_type,impulse_duty_cycle);
    
    INFO = struct(...
        'enunt',['Sa se identifice modelul matematic al procesului industrial descris ' ...
        'prin comportamentul intrare/iesire u(t)/y(t) utilizand metoda tangentei. ' ...
        'Semnalele sunt achizitionate cu un osciloscop si scalate, iar sistemul de achizitie de date nu are ecranare corespunzatoare.'],...
        'mentiuni_suplimentare','Eroarea de identificare sa fie sub 12%.'...
        );
end

%%
function [t,u,y,INFO] = gen_sub_ord_sup_met_cohen_coon_zgomot_impulsiv(m_sub,n_sub)
% Proces industrial (ordin superior), metoda Cohen-Coon
% VALIDAT
    
    rng(m_sub+10*n_sub)
    K = 2.2+rand(1)*2;
    T1 = 20+50*rand(1);
    H = tf(K,[T1,1]);

    T_aux = T1*(1+1.8*(rand(1)-0.5)); % intre 20% si 180% din T1.
    H = series(H,tf(1,[T_aux,1]));

    rng(m_sub+10*n_sub)
    r = rand(1);  % posibil ordin 3
    if r < 0.5
        T_aux = T1*(1+1.9*(rand(1)-0.5));
        H = series(H,tf(1,[T_aux,1]));
    end

    r = rand(1);  % posibil ordin 4
    if r < 0.5
        T_aux = T1*(1+1.9*(rand(1)-0.5));
        H = series(H,tf(1,[T_aux,1]));
    end

    r = rand(1);  % posibil ordin 5
    if r < 0.5
        T_aux = T1*(1+1.9*(rand(1)-0.5));
        H = series(H,tf(1,[T_aux,1]));
    end

    u0 = -3+4*rand(1);
    ust = u0+1+5*rand(1);
    N = 2;
    %
    Taq_multiplier = round(400+500*rand(1));
    tol_extra_ts = 1.3+0.6*rand(1);
    tol_cut = 0.08+0.06*rand(1);
    quantizer_relerr_bits = 6;
    %
    with_noise = true;
    noise_rel_tol = 0.03;
    %
    with_imp_noise = true;
    tol_impulse = 0.01/3/4;
    rel_ampl_impulse = 1/4;
    %
    step_type = true;
    if mod(n_sub,2)==0
        reverse = false;
    else
        reverse = true;
    end
    impulse_duty_cycle = 0.04+rand(1)*0.06;
    
    [t,u,y] = gen_dataset_fcn_neparametrice(...
        H,m_sub,n_sub,u0,ust,N,...
        Taq_multiplier,tol_extra_ts,tol_cut,...
        reverse,quantizer_relerr_bits,with_noise,noise_rel_tol,with_imp_noise,...
        tol_impulse,rel_ampl_impulse,step_type,impulse_duty_cycle);
    
    INFO = struct(...
        'enunt',['Sa se identifice modelul matematic al procesului industrial descris ' ...
        'prin comportamentul intrare/iesire u(t)/y(t) utilizand metoda Cohen-Coon. ' ...
        'Semnalele sunt achizitionate cu un osciloscop si scalate, iar sistemul de achizitie de date nu are ecranare corespunzatoare.'],...
        'mentiuni_suplimentare','Eroarea de identificare sa fie sub 12%.'...
        );
end

%%
function [t,u,y,INFO] = gen_sub_ord_sup_met_neparam_treapta_zgomot_impulsiv(m_sub,n_sub)
% Proces industrial (ordin superior), metoda neparametrica nespecificata.
% Ar trebui metoda tangentei pentru ordin superior sau Cohen-Coon.
% VALIDAT
    
    rng(m_sub+10*n_sub)
    K = 2.2+rand(1)*2;
    T1 = 20+50*rand(1);
    H = tf(K,[T1,1]);

    T_aux = T1*(1+1.8*(rand(1)-0.5)); % intre 20% si 180% din T1.
    H = series(H,tf(1,[T_aux,1]));

    rng(m_sub+10*n_sub)
    r = rand(1);  % posibil ordin 3
    if r < 0.5
        T_aux = T1*(1+1.9*(rand(1)-0.5));
        H = series(H,tf(1,[T_aux,1]));
    end

    r = rand(1);  % posibil ordin 4
    if r < 0.5
        T_aux = T1*(1+1.9*(rand(1)-0.5));
        H = series(H,tf(1,[T_aux,1]));
    end

    r = rand(1);  % posibil ordin 5
    if r < 0.5
        T_aux = T1*(1+1.9*(rand(1)-0.5));
        H = series(H,tf(1,[T_aux,1]));
    end

    u0 = 1+12*rand(1);
    ust = u0+2+6*rand(1);
    N = 2;
    %
    Taq_multiplier = round(200+800*rand(1));
    tol_extra_ts = 1+0.6*rand(1);
    tol_cut = 0.08+0.08*rand(1);
    quantizer_relerr_bits = 6;
    %
    with_noise = true;
    noise_rel_tol = 0.03;
    %
    with_imp_noise = true;
    tol_impulse = 0.01/3/4;
    rel_ampl_impulse = 1/4;
    %
    step_type = true;
    if mod(n_sub,2)==0
        reverse = false;
    else
        reverse = true;
    end
    impulse_duty_cycle = 0.04+rand(1)*0.06;
    
    [t,u,y] = gen_dataset_fcn_neparametrice(...
        H,m_sub,n_sub,u0,ust,N,...
        Taq_multiplier,tol_extra_ts,tol_cut,...
        reverse,quantizer_relerr_bits,with_noise,noise_rel_tol,with_imp_noise,...
        tol_impulse,rel_ampl_impulse,step_type,impulse_duty_cycle);
    
    INFO = struct(...
        'enunt',['Sa se identifice modelul matematic al procesului industrial descris ' ...
        'prin comportamentul intrare/iesire u(t)/y(t) utilizand o metoda neparametrica. ' ...
        'Semnalele sunt achizitionate cu un osciloscop si scalate, iar sistemul de achizitie de date nu are ecranare corespunzatoare.'],...
        'mentiuni_suplimentare','Eroarea de identificare sa fie sub 12%.'...
        );
end

%% SUBIECTE METODE PARAMETRICE
function [t,u,y,INFO] = gen_sub_met_param_sistem_mecanic_doua_zerouri_trei_poli(m_sub,n_sub,tip_subiect)
    % Sistem mecanic cu doua zerouri si trei poli pentru m=2 
    % Sistem mecanic cu trei poli fara zerouri    pentru m=0
    
    rng(n_sub)
    % calibrare SPAB
    N = 5;
    bi = 3;
    bj = 5;
    
    p = 100;

    assert(m_sub==0 || m_sub==2)
    
    T1 = 5*n_sub;
    T2 = 3*n_sub;
    T3 = 1*n_sub;
    T1_z = 10*m_sub;
    T2_z = 20*m_sub;
    K = n_sub^2;

    H = K*tf(conv([T1_z 1],[T2_z 1]),conv(conv([T1,1],[T2,1]),[T3,1]));
    
    u0 = -500+300*rand(1);
    ust = u0+1000*rand(1);
    num_steps = 3;
    
    %
    Taq_multiplier = round(500+300*rand(1));
    tol_extra_ts = 1.2+0.5*rand(1);
    tol_cut = 0.08+0.05*rand(1);
    quantizer_relerr_bits = 6;

    factor_decimare = round(1+7*rand(1));
    
    with_noise = true;
    noise_rel_tol = 0.03;
    
    if strcmp(tip_subiect,'xcorr')
        with_input_noise = false;
    else
        with_input_noise = true;
    end
    [t,u,y] = gen_dataset_fcn_parametrice(H,m_sub,n_sub,...
                u0,ust,N,bi,bj,p,Taq_multiplier,tol_extra_ts,tol_cut,...
                quantizer_relerr_bits,with_noise,noise_rel_tol,...
                with_input_noise,num_steps,factor_decimare);
    
    INFO = struct(...
        'enunt','',...
        'mentiuni_suplimentare','');
    INFO.enunt = ['Un sistem mecanic de rotație cu m' ...
                ' zerouri are ca intrare forța aplicată a F și are montat' ...
                ' un senzor care măsoară turația w a elementului aflat' ...
                ' în mișcare de rotație. În urma unui experiment efectuat' ...
                ' asupra acestui proces au rezultat datele:' ...
                ' t - timpul [s], u - forța [N] și y - turația [rot/min].'];

    if strcmp(tip_subiect,'corr')
        INFO.mentiuni_suplimentare = ['Să se realizeze o identificare' ...
                ' a modelului procesului folosind o metodă parametrică.' ...
                ' Se cere ca modelul intrare–ieșire să fie validat' ...
                ' prin autocorelație, iar eroarea cu care se realizează' ...
                ' identificarea modelului să fie sub 5%.'];
    elseif strcmp(tip_subiect,'xcorr')
        INFO.mentiuni_suplimentare = ['Să se realizeze o identificare' ...
                ' a modelului procesului folosind o metodă parametrică.' ...
                ' Se cere ca modelul intrare–ieșire să fie validat' ...
                ' prin intercorelație, iar eroarea cu care se realizează' ...
                ' identificarea modelului să fie sub 5%.'];
    elseif strcmp(tip_subiect,'ss')
        INFO.mentiuni_suplimentare = ['Să se realizeze o identificare' ...
                ' a modelului procesului folosind o metodă parametrică.' ...
                ' Se cere ca modelul de tip spațiul stărilor să fie validat, ' ...
                ' iar eroarea cu care se realizează' ...
                ' identificarea modelului să fie sub 5%.'];
    else
        error('Mai verificati o data tipul subiectului!')
    end
end

%%
function [t,u,y,INFO] = gen_sub_met_param_sistem_mecanic_un_zero(m_sub,n_sub,tip_subiect)
    % Sistem mecanic cu un zero si doi poli   pentru m = 1
    % Sistem mecanic cu doi poli fara zerouri pentru m = 0

    rng(n_sub)
    % calibrare SPAB
    N = 5;
    bi = 3;
    bj = 5;
    
    p = 100;

    assert(m_sub==0 || m_sub==1)
    
    T1 = 10*n_sub;
    T2 = 20*n_sub;
    T1_z = 100*m_sub;
    K = 5*n_sub^3;
    
    H = K*tf([T1_z 1],conv([T1,1],[T2,1]));
    
    u0 = -200+400*rand(1);
    ust = u0+2000*rand(1);
    num_steps = 3;
    
    %
    Taq_multiplier = round(500+300*rand(1));
    tol_extra_ts = 1.2+0.5*rand(1);
    tol_cut = 0.08+0.05*rand(1);
    quantizer_relerr_bits = 6;

    factor_decimare = round(3+9*rand(1));
    
    with_noise = true;
    noise_rel_tol = 0.03;
    if strcmp(tip_subiect,'xcorr')
        with_input_noise = false;
    else
        with_input_noise = true;
    end
    [t,u,y] = gen_dataset_fcn_parametrice(H,m_sub,n_sub,...
                u0,ust,N,bi,bj,p,Taq_multiplier,tol_extra_ts,tol_cut,...
                quantizer_relerr_bits,with_noise,noise_rel_tol,...
                with_input_noise,num_steps,factor_decimare);
    
    INFO = struct(...
        'enunt','',...
        'mentiuni_suplimentare','');
    INFO.enunt = ['Un sistem mecanic de rotație cu m' ...
                ' zerouri are ca intrare forța aplicată a F și are montat' ...
                ' un senzor care măsoară turația w a elementului aflat' ...
                ' în mișcare de rotație. În urma unui experiment efectuat' ...
                ' asupra acestui proces au rezultat datele:' ...
                ' t - timpul [s], u - forța [N] și y - turația [rot/min].'];

    if strcmp(tip_subiect,'corr')
        INFO.mentiuni_suplimentare = ['Să se realizeze o identificare' ...
                ' a modelului procesului folosind o metodă parametrică.' ...
                ' Se cere ca modelul intrare–ieșire să fie validat' ...
                ' prin autocorelație, iar eroarea cu care se realizează' ...
                ' identificarea modelului să fie sub 5%.'];
    elseif strcmp(tip_subiect,'xcorr')
        INFO.mentiuni_suplimentare = ['Să se realizeze o identificare' ...
                ' a modelului procesului folosind o metodă parametrică.' ...
                ' Se cere ca modelul intrare–ieșire să fie validat' ...
                ' prin intercorelație, iar eroarea cu care se realizează' ...
                ' identificarea modelului să fie sub 5%.'];
    elseif strcmp(tip_subiect,'ss')
        INFO.mentiuni_suplimentare = ['Să se realizeze o identificare' ...
                ' a modelului procesului folosind o metodă parametrică.' ...
                ' Se cere ca modelul de tip spațiul stărilor să fie validat, ' ...
                ' iar eroarea cu care se realizează' ...
                ' identificarea modelului să fie sub 5%.'];
    else
        error('Mai verificati o data tipul subiectului!')
    end

end

%%
function [t,u,y,INFO] = gen_sub_met_param_sistem_doua_rezervoare(m_sub,n_sub,tip_subiect)
    % Sistem cu doua rezervoare, doi poli, fara zerouri
    rng(n_sub)
    % calibrare SPAB
    N = 5;
    bi = 3;
    bj = 5;
    
    p = 50;
    
    T1 = 30+n_sub*rand(1);
    T2 = 50+m_sub*rand(1);
    K = n_sub; 
    
    A = [0 1; -1/T1/T2 -1/T1-1/T2];
    B = [0; K/T1/T2];
    C = [1 0];
    D = 0;
    
    H = ss(A,B,C,D);
    
    u0 = 30+5*rand(1);
    ust = u0+20*rand(1);
    num_steps = 4;
    
    %
    Taq_multiplier = round(500+300*rand(1));
    tol_extra_ts = 1.2+0.5*rand(1);
    tol_cut = 0.08+0.05*rand(1);
    quantizer_relerr_bits = 6;

    factor_decimare = round(1 + 8*rand(1));
    
    with_noise = true;
    noise_rel_tol = 0.03;
    if strcmp(tip_subiect,'xcorr')
        with_input_noise = false;
    else
        with_input_noise = true;
    end
    [t,u,y] = gen_dataset_fcn_parametrice(H,m_sub,n_sub,...
                u0,ust,N,bi,bj,p,Taq_multiplier,tol_extra_ts,tol_cut,...
                quantizer_relerr_bits,with_noise,noise_rel_tol,...
                with_input_noise,num_steps,factor_decimare);
    
    INFO = struct(...
        'enunt','',...
        'mentiuni_suplimentare','');
    INFO.enunt = ['Un sistem format din două rezervoare cu apă înseriate ' ...
        'este alimentat cu un debit de lichid Q_{in} și are montat un' ...
        ' senzor care măsoară debitul de ieșire Q_{out} din cel de-al ' ...
        ' doilea rezervor. În urma unui experiment efectuat asupra acestui' ...
        ' proces au rezultat datele: t - timpul [s], u - debitul de intrare' ...
        ' [m^3/h] și y - debitul de ieșire [m^3/h].'];

    if strcmp(tip_subiect,'corr')
        INFO.mentiuni_suplimentare = ['Să se realizeze o identificare' ...
                ' a modelului procesului folosind o metodă parametrică.' ...
                ' Se cere ca modelul intrare–ieșire să fie validat' ...
                ' prin autocorelație, iar eroarea cu care se realizează' ...
                ' identificarea modelului să fie sub 5%.'];
    elseif strcmp(tip_subiect,'xcorr')
        INFO.mentiuni_suplimentare = ['Să se realizeze o identificare' ...
                ' a modelului procesului folosind o metodă parametrică.' ...
                ' Se cere ca modelul intrare–ieșire să fie validat' ...
                ' prin intercorelație, iar eroarea cu care se realizează' ...
                ' identificarea modelului să fie sub 5%.'];
    elseif strcmp(tip_subiect,'ss')
        INFO.mentiuni_suplimentare = ['Să se realizeze o identificare' ...
                ' a modelului procesului folosind o metodă parametrică.' ...
                ' Se cere ca modelul de tip spațiul stărilor să fie validat, ' ...
                ' iar eroarea cu care se realizează' ...
                ' identificarea modelului să fie sub 5%.'];
    else
        error('Mai verificati o data tipul subiectului!')
    end

end

%%
function [t,u,y,INFO] = gen_sub_met_param_sistem_ordin_trei(m_sub,n_sub,tip_subiect)
    % Sistem mecanic cu un trei poli: unul real, ceilalti doi sunt 
    % reali pentru n_sub >=3 si cc pentur n_sub < 3
    % si un zero de faza neminima pentru m_sub > 0 sau fara zerouri pentru
    % m_sub = 0

    rng(n_sub)
    % calibrare SPAB
    N = 5;
    bi = 3;
    bj = 5;
    
    p = 50;
    
    T3 = 10*n_sub;
    zeta = n_sub/3;
    wn = n_sub^2+m_sub*rand(1);
    T1_z = -2*m_sub;
    K = n_sub^2*rand(1);

    H = K*wn^2*tf([T1_z 1],conv([1 2*zeta*wn wn^2],[T3,1]));

    
    u0 = 30+5*rand(1);
    ust = u0+20*rand(1);
    num_steps = 4;
    
    %
    Taq_multiplier = round(500+300*rand(1));
    tol_extra_ts = 1.2+0.5*rand(1);
    tol_cut = 0.08+0.05*rand(1);
    quantizer_relerr_bits = 6;

    factor_decimare = round(1 + 8*rand(1));
    
    with_noise = true;
    noise_rel_tol = 0.03;
    if strcmp(tip_subiect,'xcorr')
        with_input_noise = false;
    else
        with_input_noise = true;
    end
    [t,u,y] = gen_dataset_fcn_parametrice(H,m_sub,n_sub,...
                u0,ust,N,bi,bj,p,Taq_multiplier,tol_extra_ts,tol_cut,...
                quantizer_relerr_bits,with_noise,noise_rel_tol,...
                with_input_noise,num_steps,factor_decimare);
    
    INFO = struct(...
        'enunt','',...
        'mentiuni_suplimentare','');
    if m_sub > 0
        INFO.enunt = ['Un circuit electric de putere având un zero de fază' ...
        ' neminimă are ca intrare tensiunea' ...
        ' electrică V_{in} și are montat un senzor care măsoară curentul' ...
        ' printr-o rezistență de sarcină I_{out}. În urma unui experiment' ...
        ' efectuat asupra acestui proces au rezultat datele: t - timpul [s], ' ...
        ' u - tensiunea de intrare [V] și y - curentul de sarcină [A]'];
    else
        INFO.enunt = ['Un circuit electric de putere are ca intrare tensiunea' ...
        ' electrică V_{in} și are montat un senzor care măsoară curentul' ...
        ' printr-o rezistență de sarcină I_{out}. În urma unui experiment' ...
        ' efectuat asupra acestui proces au rezultat datele: t - timpul [s], ' ...
        ' u - tensiunea de intrare [V] și y - curentul de sarcină [A]'];
    end
    if strcmp(tip_subiect,'corr')
        INFO.mentiuni_suplimentare = ['Să se realizeze o identificare' ...
                ' a modelului procesului folosind o metodă parametrică.' ...
                ' Se cere ca modelul intrare–ieșire să fie validat' ...
                ' prin autocorelație, iar eroarea cu care se realizează' ...
                ' identificarea modelului să fie sub 5%.'];
    elseif strcmp(tip_subiect,'xcorr')
        INFO.mentiuni_suplimentare = ['Să se realizeze o identificare' ...
                ' a modelului procesului folosind o metodă parametrică.' ...
                ' Se cere ca modelul intrare–ieșire să fie validat' ...
                ' prin intercorelație, iar eroarea cu care se realizează' ...
                ' identificarea modelului să fie sub 5%.'];
    elseif strcmp(tip_subiect,'ss')
        INFO.mentiuni_suplimentare = ['Să se realizeze o identificare' ...
                ' a modelului procesului folosind o metodă parametrică.' ...
                ' Se cere ca modelul de tip spațiul stărilor să fie validat, ' ...
                ' iar eroarea cu care se realizează' ...
                ' identificarea modelului să fie sub 5%.'];
    else
        error('Mai verificati o data tipul subiectului!')
    end

end