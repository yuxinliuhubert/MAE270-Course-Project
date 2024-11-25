%% Clean workspace

clear all; clf; close all; clc;

%% Data Preparation

% Section 1 - Load Data:
% Load the data provided for each input channel.

% Sampling parameters
ts = 1/50; % Sample period in seconds
fs = 1/ts; % Sampling frequency in Hz

% Load data for u1
load('random_u1.mat'); % Contains u1, y1, y2
u1 = u1;
y11 = y1; % Output y1 due to input u1
y21 = y2; % Output y2 due to input u1

% Load data for u2
load('random_u2.mat'); % Contains u2, y1, y2
u2 = u2;
y12 = y1; % Output y1 due to input u2
y22 = y2; % Output y2 due to input u2

% Load data for u3
load('random_u3.mat'); % Contains u3, y1, y2
u3 = u3;
y13 = y1; % Output y1 due to input u3
y23 = y2; % Output y2 due to input u3

% Time vector
Ndat = length(u1); % Assuming all datasets have the same length
t = (0:Ndat-1) * ts;

% Verification: Check if all datasets have the same length
if length(u1) ~= length(u2) || length(u1) ~= length(u3)
    error('Input signals u1, u2, and u3 do not have the same length.');
end

disp('Data loaded successfully. All input signals have the same length.');
disp(['Number of data points: ', num2str(Ndat)]);
disp('----------------------------------------');

% Section 2 -  Code for Plotting:
% Plot the input and output signals to get an initial understanding of the system's behavior.

% Plot for u1
figure;
subplot(3,1,1);
plot(t, u1);
title('Input u_1');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

subplot(3,1,2);
plot(t, y11);
title('Output y_1 due to u_1');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

subplot(3,1,3);
plot(t, y21);
title('Output y_2 due to u_1');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

% Verification: Check basic statistics of signals
disp('Basic statistics of u1:');
disp(['Mean: ', num2str(mean(u1)), ', Std Dev: ', num2str(std(u1))]);
disp('Basic statistics of y11:');
disp(['Mean: ', num2str(mean(y11)), ', Std Dev: ', num2str(std(y11))]);
disp('Basic statistics of y21:');
disp(['Mean: ', num2str(mean(y21)), ', Std Dev: ', num2str(std(y21))]);
disp('----------------------------------------');

%% Task 1 - Empirical Frequency Response Estimates

% Section 1 - Compute Spectra Using cpsd:

% Parameters for cpsd
nfft = 1024; % Number of FFT points
window = hamming(nfft);
noverlap = []; % Default overlap

% Compute auto-spectra
[Suu1, f] = cpsd(u1, u1, window, noverlap, nfft, fs, 'twosided');
[Suu2, ~] = cpsd(u2, u2, window, noverlap, nfft, fs, 'twosided');
[Suu3, ~] = cpsd(u3, u3, window, noverlap, nfft, fs, 'twosided');

% Compute cross-spectra
[Su1u2, ~] = cpsd(u1, u2, window, noverlap, nfft, fs, 'twosided');
[Su1u3, ~] = cpsd(u1, u3, window, noverlap, nfft, fs, 'twosided');
[Su2u3, ~] = cpsd(u2, u3, window, noverlap, nfft, fs, 'twosided');

% Verification: Check sizes of spectral densities
disp('Size of Suu1:');
disp(size(Suu1));
disp('Size of frequency vector f:');
disp(size(f));
disp('----------------------------------------');

% Section 2 - Plotting Auto-Spectra:

% Plot auto-spectra
figure;
loglog(f, abs(Suu1), 'r', f, abs(Suu2), 'g', f, abs(Suu3), 'b');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
title('Auto-Spectral Densities of Inputs');
legend('S_{u_1u_1}', 'S_{u_2u_2}', 'S_{u_3u_3}');
axis([0.1 fs/2 1e-5 1e-2]);
grid on;

% Section 3 - Plotting Cross-Spectra:

% Plot cross-spectra
figure;
loglog(f, abs(Su1u2), 'r', f, abs(Su1u3), 'g', f, abs(Su2u3), 'b');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
title('Cross-Spectral Densities between Inputs');
legend('S_{u_1u_2}', 'S_{u_1u_3}', 'S_{u_2u_3}');
axis([0.1 fs/2 1e-5 1e-2]);
grid on;

% Verification: Check maximum cross-spectral densities
disp('Maximum magnitude of cross-spectral density Su1u2:');
disp(max(abs(Su1u2)));
disp('Maximum magnitude of Suu1:');
disp(max(abs(Suu1)));
disp('----------------------------------------');

% Section 4 - Calculating Mean Square Values:

% Compute the variance (mean square value) of each input signal in the time domain:
var_u1 = mean(u1.^2);
var_u2 = mean(u2.^2);
var_u3 = mean(u3.^2);

% Compute the mean of the auto-spectral densities and multiply by the sampling frequency:
mean_Suu1 = mean(abs(Suu1)) * fs;
mean_Suu2 = mean(abs(Suu2)) * fs;
mean_Suu3 = mean(abs(Suu3)) * fs;

% Verification: Compare time and frequency domain variances
disp('Variance of u1 (time domain):');
disp(var_u1);
disp('Variance of u1 (frequency domain):');
disp(mean_Suu1);
difference = abs(var_u1 - mean_Suu1);
disp('Difference between time and frequency domain variances for u1:');
disp(difference);
disp('----------------------------------------');
% Repeat for u2
disp('Variance of u2 (time domain):');
disp(var_u2);
disp('Variance of u2 (frequency domain):');
disp(mean_Suu2);
difference = abs(var_u2 - mean_Suu2);
disp('Difference between time and frequency domain variances for u2:');
disp(difference);
disp('----------------------------------------');
% Repeat for u3
disp('Variance of u3 (time domain):');
disp(var_u3);
disp('Variance of u3 (frequency domain):');
disp(mean_Suu3);
difference = abs(var_u3 - mean_Suu3);
disp('Difference between time and frequency domain variances for u3:');
disp(difference);
disp('----------------------------------------');

% Section 5 - Estimating Frequency Responses

% For each input-output pair, compute the frequency response:

% For input u1
[Sy1u1, ~] = cpsd(y11, u1, window, noverlap, nfft, fs, 'twosided');
[Sy2u1, ~] = cpsd(y21, u1, window, noverlap, nfft, fs, 'twosided');

H11 = Sy1u1 ./ Suu1;
H21 = Sy2u1 ./ Suu1;

% For input u2
[Sy1u2, ~] = cpsd(y12, u2, window, noverlap, nfft, fs, 'twosided');
[Sy2u2, ~] = cpsd(y22, u2, window, noverlap, nfft, fs, 'twosided');

H12 = Sy1u2 ./ Suu2;
H22 = Sy2u2 ./ Suu2;

% For input u3
[Sy1u3, ~] = cpsd(y13, u3, window, noverlap, nfft, fs, 'twosided');
[Sy2u3, ~] = cpsd(y23, u3, window, noverlap, nfft, fs, 'twosided');

H13 = Sy1u3 ./ Suu3;
H23 = Sy2u3 ./ Suu3;

% Verification: Check sizes of frequency responses
disp('Size of H11:');
disp(size(H11));
disp('Size of H21:');
disp(size(H21));
disp('Size of frequency vector f:');
disp(size(f));
disp('----------------------------------------');

% Section 6 - Plotting Frequency Responses

% Magnitude Plots:

% Magnitude of H11 and H21
figure;
loglog(f, abs(H11), 'r', f, abs(H21), 'b');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
title('Frequency Response Magnitudes for Input u_1');
legend('H_{11}', 'H_{21}');
axis([0.1 fs/2 1e-3 1e2]);
grid on;

% Magnitude for H12 and H22
figure;
loglog(f, abs(H12), 'r', f, abs(H22), 'b');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
title('Frequency Response Magnitudes for Input u_2');
legend('H_{12}', 'H_{22}');
axis([0.1 fs/2 1e-3 1e2]);
grid on;

% Magnitude for H13 and H23
figure;
loglog(f, abs(H13), 'r', f, abs(H23), 'b');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
title('Frequency Response Magnitudes for Input u_3');
legend('H_{13}', 'H_{23}');
axis([0.1 fs/2 1e-3 1e2]);
grid on;

% Phase Plots:
% Phase of H11 and H21
figure;
semilogx(f, angle(H11)*(180/pi), 'r', f, angle(H21)*(180/pi), 'b');
xlabel('Frequency (Hz)');
ylabel('Phase (degrees)');
title('Frequency Response Phases for Input u_1');
legend('H_{11}', 'H_{21}');
axis([0.1 fs/2 -200 200]);
grid on;

% Phase for H12 and H22
figure;
semilogx(f, angle(H12)*(180/pi), 'r', f, angle(H22)*(180/pi), 'b');
xlabel('Frequency (Hz)');
ylabel('Phase (degrees)');
title('Frequency Response Phases for Input u_2');
legend('H_{12}', 'H_{22}');
axis([0.1 fs/2 -200 200]);
grid on;

% Phase for H13 and H23
figure;
semilogx(f, angle(H13)*(180/pi), 'r', f, angle(H23)*(180/pi), 'b');
xlabel('Frequency (Hz)');
ylabel('Phase (degrees)');
title('Frequency Response Phases for Input u_3');
legend('H_{13}', 'H_{23}');
axis([0.1 fs/2 -200 200]);
grid on;

% Verification: Plot real and imaginary parts of H11
figure;
subplot(2,1,1);
plot(f, real(H11));
title('Real Part of H_{11}');
xlabel('Frequency (Hz)');
ylabel('Real(H_{11})');
grid on;

subplot(2,1,2);
plot(f, imag(H11));
title('Imaginary Part of H_{11}');
xlabel('Frequency (Hz)');
ylabel('Imag(H_{11})');
grid on;

disp('Task 1 completed successfully.');
disp('----------------------------------------');

%% Task 2 - Pulse Response Estimates

% Section 1 - Compute Pulse Responses using IFFT

% Compute the IFFT of the frequency responses to obtain pulse responses
h11 = ifft(H11);
h21 = ifft(H21);

h12 = ifft(H12);
h22 = ifft(H22);

h13 = ifft(H13);
h23 = ifft(H23);

% Time vector for pulse responses
n_pulse = length(h11); % Should be equal to nfft
t_pulse = (0:n_pulse-1) * ts; % Starts at t=0

% Verification: Check that n_pulse equals nfft
if n_pulse ~= nfft
    error('Length of pulse responses does not equal nfft.');
end

disp('Pulse responses computed successfully.');
disp(['Length of pulse responses: ', num2str(n_pulse)]);
disp('----------------------------------------');

% Section 2 - Plotting Pulse Responses

% Plot h11 and h21
figure;
subplot(2,1,1);
plot(t_pulse, real(h11), 'r');
xlabel('Time (s)');
ylabel('Amplitude');
title('Pulse Response h_{11}');
axis([0 3 -2 3]);
grid on;

subplot(2,1,2);
plot(t_pulse, real(h21), 'b');
xlabel('Time (s)');
ylabel('Amplitude');
title('Pulse Response h_{21}');
axis([0 3 -2 3]);
grid on;

% Plot h12 and h22
figure;
subplot(2,1,1);
plot(t_pulse, real(h12), 'r');
xlabel('Time (s)');
ylabel('Amplitude');
title('Pulse Response h_{12}');
axis([0 3 -2 3]);
grid on;

subplot(2,1,2);
plot(t_pulse, real(h22), 'b');
xlabel('Time (s)');
ylabel('Amplitude');
title('Pulse Response h_{22}');
axis([0 3 -2 3]);
grid on;

% Plot h13 and h23
figure;
subplot(2,1,1);
plot(t_pulse, real(h13), 'r');
xlabel('Time (s)');
ylabel('Amplitude');
title('Pulse Response h_{13}');
axis([0 3 -2 3]);
grid on;

subplot(2,1,2);
plot(t_pulse, real(h23), 'b');
xlabel('Time (s)');
ylabel('Amplitude');
title('Pulse Response h_{23}');
axis([0 3 -2 3]);
grid on;

% Check maximum imaginary part (should be negligible)
max_imag_h11 = max(abs(imag(h11)));
disp(['Maximum imaginary part of h11: ', num2str(max_imag_h11)]);

% Repeat for other pulse responses
max_imag_h21 = max(abs(imag(h21)));
disp(['Maximum imaginary part of h21: ', num2str(max_imag_h21)]);

max_imag_h12 = max(abs(imag(h12)));
disp(['Maximum imaginary part of h12: ', num2str(max_imag_h12)]);

max_imag_h22 = max(abs(imag(h22)));
disp(['Maximum imaginary part of h22: ', num2str(max_imag_h22)]);

max_imag_h13 = max(abs(imag(h13)));
disp(['Maximum imaginary part of h13: ', num2str(max_imag_h13)]);

max_imag_h23 = max(abs(imag(h23)));
disp(['Maximum imaginary part of h23: ', num2str(max_imag_h23)]);

disp('----------------------------------------');

% Section 3 - Validate Parseval's Theorem

% Number of points
N = length(H11);

% Compute energy in time domain for h11
energy_time = sum(abs(h11).^2) * ts;

% Compute energy in frequency domain for H11
energy_freq = (1/N) * sum(abs(H11).^2) * ts;

% Display energies
disp(['Energy of h11 in time domain: ', num2str(energy_time)]);
disp(['Energy of H11 in frequency domain: ', num2str(energy_freq)]);
disp(['Difference in energy for H11: ', num2str(abs(energy_time - energy_freq))]);

disp('----------------------------------------');

% Repeat the energy verification for other pulse responses (optional)
% Example for h21 and H21
% Compute energy in time domain for h21
% energy_time_h21 = sum(abs(h21).^2) * ts;

% Compute energy in frequency domain for H21
% energy_freq_h21 = (1/N) * sum(abs(H21).^2) * ts;

% Display energies for h21
% disp(['Energy of h21 in time domain: ', num2str(energy_time_h21)]);
% disp(['Energy of H21 in frequency domain: ', num2str(energy_freq_h21)]);
% disp(['Difference in energy for H21: ', num2str(abs(energy_time_h21 - energy_freq_h21))]);

% disp('----------------------------------------');

disp('Task 2 completed successfully.');
disp('----------------------------------------');


%% Task 3 - Hankel Matrix Analysis and Parametric Model

% Section 1 - Construct the Hankel Matrix M_n

% Number of samples to use for constructing M_n
n = 25; % As per the project instructions
K = 2*n - 1; % Number of required pulse response samples

% Ensure that K does not exceed the length of h11
if K > length(h11)
    error('Not enough data points in h11 to construct M_n with n = %d', n);
end

% Initialize variables
m = 2; % Number of outputs
q = 3; % Number of inputs

% Prepare h[k] as a cell array
h = cell(K, 1);
for k = 1:K
    % At each time k, h[k] is an m x q matrix
    % Collect h[k] from h11, h12, h13, h21, h22, h23
    h_k = [h11(k), h12(k), h13(k);  % First row for output y1
           h21(k), h22(k), h23(k)]; % Second row for output y2
    h{k} = h_k;
end

% Now construct the Hankel matrix M_n

% Preallocate M_n
M_n = zeros(m * n, q * n);

% Fill M_n
for i = 1:n
    for j = 1:n
        idx = i + j - 1; % Index in h, starting from 1
        h_ij = h{idx};
        row_idx = (i-1)*m + (1:m);
        col_idx = (j-1)*q + (1:q);
        M_n(row_idx, col_idx) = h_ij;
    end
end

% Verification: Check dimensions of M_n
disp('Size of Hankel matrix M_n:');
disp(size(M_n));
expected_rows = m * n;
expected_cols = q * n;
disp(['Expected size: ', num2str(expected_rows), ' rows x ', num2str(expected_cols), ' columns']);
disp('----------------------------------------');

% Section 2 - Perform SVD on M_n

[U, S, V] = svd(M_n);

% Plot singular values
singular_values = diag(S);
figure;
semilogy(singular_values, 'o-');
xlabel('Index');
ylabel('Singular Value (log scale)');
title('Singular Values of M_n');
grid on;

% Verification: Print singular values
disp('Singular values of M_n:');
disp(singular_values');
disp('----------------------------------------');

% Verify orthonormality of U and V
identity_U = U' * U;
identity_V = V' * V;
deviation_U = norm(identity_U - eye(size(identity_U)));
deviation_V = norm(identity_V - eye(size(identity_V)));
disp(['Deviation of U'' * U from identity: ', num2str(deviation_U)]);
disp(['Deviation of V'' * V from identity: ', num2str(deviation_V)]);
disp('----------------------------------------');

% Section 3 - Estimate models for different model orders

model_orders = [7, 8, 10, 16];
num_models = length(model_orders);

% Preallocate cell arrays to store models
A_models = cell(num_models, 1);
B_models = cell(num_models, 1);
C_models = cell(num_models, 1);
D_models = cell(num_models, 1); % D is assumed to be zero

% Construct shifted Hankel matrix M_n_tilde

% Preallocate M_n_tilde
M_n_tilde = zeros(m * n, q * n);

for i = 1:n
    for j = 1:n
        idx = i + j; % Shifted by +1
        if idx <= K
            h_ij = h{idx};
        else
            h_ij = zeros(m, q); % If idx exceeds K, pad with zeros
        end
        row_idx = (i-1)*m + (1:m);
        col_idx = (j-1)*q + (1:q);
        M_n_tilde(row_idx, col_idx) = h_ij;
    end
end

% For each model order, estimate the model

for idx = 1:num_models
    n_s = model_orders(idx);

    % Truncate U, S, V
    U1 = U(:, 1:n_s);
    S1 = S(1:n_s, 1:n_s);
    V1 = V(:, 1:n_s);

    % Compute L and R
    L = U1;
    R = S1 * V1';

    % Compute pseudo-inverses
    L_pinv = pinv(L);
    R_pinv = pinv(R);

    % Compute A
    A = L_pinv * M_n_tilde * R_pinv;

    % Extract C (first m rows of L)
    C = L(1:m, :);

    % Extract B (first q columns of R)
    B = R(:, 1:q);

    % Assume D = zero matrix
    D = zeros(m, q);

    % Store the model
    A_models{idx} = A;
    B_models{idx} = B;
    C_models{idx} = C;
    D_models{idx} = D;

    % Check stability
    eig_A = eig(A);
    max_abs_eig = max(abs(eig_A));
    disp(['Model order ', num2str(n_s), ': Max |eig(A)| = ', num2str(max_abs_eig)]);

    if max_abs_eig >= 1
        disp('Warning: The model is unstable.');
    end

    % Verification: Check dimensions of A, B, C, D
    disp(['Size of A (Model Order ', num2str(n_s), '):']);
    disp(size(A));
    disp(['Size of B (Model Order ', num2str(n_s), '):']);
    disp(size(B));
    disp(['Size of C (Model Order ', num2str(n_s), '):']);
    disp(size(C));
    disp(['Size of D (Model Order ', num2str(n_s), '):']);
    disp(size(D));
    disp('----------------------------------------');
end

disp('Task 3 completed successfully.');
disp('----------------------------------------');

%% Task 4 - Transmission Zeros and Eigenvalue-Zero Cancellations

% Number of models
num_models = length(model_orders);

% Preallocate cell arrays to store zeros and poles
zeros_models = cell(num_models, 1);
poles_models = cell(num_models, 1);

% Tolerance for numerical comparison
tol = 1e-4;

for idx = 1:num_models
    n_s = model_orders(idx);
    A = A_models{idx};
    B = B_models{idx};
    C = C_models{idx};
    D = D_models{idx};
    
    % Create a state-space model
    sys = ss(A, B, C, D, ts);
    
    % Compute transmission zeros
    tz = tzero(sys);
    
    % Compute poles (eigenvalues of A)
    tp = eig(A);
    
    % Store zeros and poles
    zeros_models{idx} = tz;
    poles_models{idx} = tp;
    
    % Calculate magnitudes of poles
    pole_magnitudes = abs(tp);
    
    % Display zeros and poles with additional information
    disp(['Model Order ', num2str(n_s), ':']);
    if ~isempty(tz)
        disp(['Transmission Zeros: ', num2str(tz.')]);
    else
        disp('Transmission Zeros: None');
    end
    disp(['Number of Poles: ', num2str(length(tp))]);
    disp(['Poles (Eigenvalues of A): ', num2str(tp.')]);
    disp(['Pole Magnitudes: ', num2str(pole_magnitudes.')]);
    disp(['Maximum Pole Magnitude: ', num2str(max(pole_magnitudes))]);
    disp(['Minimum Pole Magnitude: ', num2str(min(pole_magnitudes))]);
    disp('----------------------------------------');
    
    % Verification: Check minimum distance between poles and zeros
    if ~isempty(tz)
        min_distance = min(abs(tz(:) - tp(:).'));
        disp(['Minimum distance between poles and zeros for Model Order ', num2str(n_s), ': ', num2str(min_distance)]);
    end
end

% Compare zeros and poles for each model
for idx = 1:num_models
    n_s = model_orders(idx);
    tz = zeros_models{idx};
    tp = poles_models{idx};
    
    % Round zeros and poles for comparison within tolerance
    tz_rounded = round(tz / tol) * tol;
    tp_rounded = round(tp / tol) * tol;
    
    % Find common values
    [common_values, zero_idx, pole_idx] = intersect(tz_rounded, tp_rounded);
    
    % Display the results
    disp(['Model Order ', num2str(n_s), ':']);
    if ~isempty(common_values)
        disp('Eigenvalue-Zero Cancellations found at:');
        for k = 1:length(common_values)
            disp(['Zero and Pole at z = ', num2str(common_values(k))]);
        end
    else
        disp('No Eigenvalue-Zero Cancellations found.');
    end
    disp('----------------------------------------');
end

% Plot poles and zeros for each model
for idx = 1:num_models
    n_s = model_orders(idx);
    tz = zeros_models{idx};
    tp = poles_models{idx};
    
    figure;
    hold on;
    
    % Plot poles
    h_poles = plot(real(tp), imag(tp), 'bx', 'MarkerSize', 10, 'LineWidth', 2); % Poles as blue 'x'
    
    % Check if zeros exist before plotting
    if ~isempty(tz)
        h_zeros = plot(real(tz), imag(tz), 'ro', 'MarkerSize', 8, 'LineWidth', 2); % Zeros as red 'o'
        legend_handles = [h_poles; h_zeros];
        legend_entries = {'Poles', 'Zeros'};
    else
        legend_handles = h_poles;
        legend_entries = {'Poles'};
    end
    
    % Plot unit circle for reference
    theta = linspace(0, 2*pi, 300);
    unit_circle = plot(cos(theta), sin(theta), 'k--');
    
    xlabel('Real Part');
    ylabel('Imaginary Part');
    title(['Poles and Zeros in z-plane - Model Order ', num2str(n_s)]);
    legend(legend_handles, legend_entries);
    grid on;
    axis equal;
    hold off;
end

disp('Task 4 completed successfully.');
disp('----------------------------------------');

%% Task 5 - Controller Design and Closed-Loop System Analysis

% Number of models
num_models = length(model_orders);

% Desired closed-loop pole locations (example values)
desired_poles_base = [0.5 + 0.2j, 0.5 - 0.2j, 0.4 + 0.3j, 0.4 - 0.3j, 0.3 + 0.1j, 0.3 - 0.1j, 0.2];

% Preallocate cell arrays for K
K_models = cell(num_models, 1);

for idx = 1:num_models
    n_s = model_orders(idx);
    A = A_models{idx};
    B = B_models{idx};
    C = C_models{idx};
    D = D_models{idx}; % Include D for completeness
    
    % Check controllability
    Co = ctrb(A, B);
    rank_Co = rank(Co);
    
    disp(['Model Order ', num2str(n_s), ':']);
    disp(['Rank of Controllability Matrix: ', num2str(rank_Co)]);
    
    if rank_Co == n_s
        disp('The system is controllable.');
    else
        disp('The system is NOT controllable.');
        disp('Proceeding to the next model.');
        disp('----------------------------------------');
        continue;
    end
    
    % Adjust desired poles to match system order
    n_poles = n_s;
    if length(desired_poles_base) < n_poles
        % Add additional poles at 0.1
        additional_poles = 0.1 * ones(1, n_poles - length(desired_poles_base));
        poles_cl = [desired_poles_base, additional_poles];
    else
        poles_cl = desired_poles_base(1:n_poles);
    end
    
    % Compute state feedback gain K
    try
        K = place(A, B, poles_cl);
    catch ME
        disp(['Could not place poles for Model Order ', num2str(n_s)]);
        disp(ME.message);
        disp('Proceeding to the next model.');
        disp('----------------------------------------');
        continue;
    end
    
    % Store K
    K_models{idx} = K;
    
    disp(['State feedback gain K computed for Model Order ', num2str(n_s)]);
    disp('Desired Closed-Loop Poles:');
    disp(poles_cl);
    disp('----------------------------------------');
    
    % Verification: Check size of K
    disp(['Size of K (Model Order ', num2str(n_s), '):']);
    disp(size(K));
    
    % Closed-loop system matrix
    A_cl = A - B * K;
    
    % Compute closed-loop eigenvalues
    eig_A_cl = eig(A_cl);
    disp(['Closed-loop eigenvalues for Model Order ', num2str(n_s), ':']);
    disp(eig_A_cl.');
    
    % Check maximum absolute eigenvalue
    max_abs_eig_cl = max(abs(eig_A_cl));
    disp(['Maximum absolute closed-loop eigenvalue: ', num2str(max_abs_eig_cl)]);
    if max_abs_eig_cl >= 1
        disp('Warning: Closed-loop system may be unstable.');
    else
        disp('Closed-loop system is stable.');
    end
    disp('----------------------------------------');
    
    % Simulate Closed-Loop System
    
    % Simulation parameters
    num_steps = 100; % Number of time steps to simulate
    t_sim = (0:num_steps-1) * ts;
    
    % Initial state
    x0 = ones(n_s, 1); % Example initial condition
    
    % Initialize state and output vectors
    x_cl = zeros(n_s, num_steps);
    y_cl = zeros(size(C,1), num_steps);
    
    x_cl(:,1) = x0;
    
    for k = 1:num_steps-1
        % Since r[k] = 0, u[k] = -K x[k]
        u = -K * x_cl(:,k);
        
        % State update
        x_cl(:,k+1) = A_cl * x_cl(:,k);
        
        % Output
        y_cl(:,k) = C * x_cl(:,k) + D * u;
    end
    
    % Compute output at the last time step
    u_last = -K * x_cl(:,num_steps);
    y_cl(:,num_steps) = C * x_cl(:,num_steps) + D * u_last;
    
    % Verification: Check final outputs and states
    disp(['Final outputs y1 and y2 for Model Order ', num2str(n_s), ':']);
    disp(y_cl(:, end));
    disp(['Final state x for Model Order ', num2str(n_s), ':']);
    disp(x_cl(:, end));
    disp('----------------------------------------');
    
    % Plot the closed-loop outputs
    figure;
    plot(t_sim, y_cl(1,:), 'r', t_sim, y_cl(2,:), 'b');
    xlabel('Time (s)');
    ylabel('Output');
    title(['Closed-Loop System Outputs - Model Order ', num2str(n_s)]);
    legend('y_1', 'y_2');
    grid on;
end

disp('Task 5 completed successfully.');
disp('----------------------------------------');
