# import necessary libraries and set up the grid
import matplotlib.pyplot as plt
import numpy as np
import random as rd
from scipy.fft import fft
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

def fit_periodic_function_and_visualize(x, y):
    from scipy.fft import fft
    from scipy.signal import find_peaks
    from scipy.optimize import curve_fit
    import numpy as np
    import matplotlib.pyplot as plt

    # 执行快速傅里叶变换
    y_fft = fft(y)
    amplitudes = np.abs(y_fft)
    half_range = len(amplitudes) // 2
    frequencies = np.arange(half_range)

    # 找到最大峰值对应的频率
    peaks, _ = find_peaks(amplitudes[:half_range])
    if len(peaks) == 0:
        # 如果没有找到峰值，使用数据长度的一部分作为默认周期
        peak_freq = 1 / (len(y) / 4)  # 默认周期为数据长度的四分之一
    else:
        peak_freq = frequencies[peaks[np.argmax(amplitudes[peaks])]]

    # 计算周期
    period = len(y) / peak_freq

    # 定义周期函数模型
    def sine_wave(x, A, phase, B):
        return A * np.sin(2 * np.pi * x / period + phase) + B

    # 拟合正弦波到数据
    params, _ = curve_fit(sine_wave, x, y, p0=[np.std(y), 0, np.mean(y)])

    # 绘制原始数据和拟合曲线
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, label='Original Data')
    plt.plot(x, sine_wave(x, *params), label='Fitted Sine Wave', color='red')
    plt.title('Original Data and Fitted Sine Wave')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

    # 打印拟合后的公式
    A, phase, B = params
    latex_formula = f"y = {A:.2f} \\cdot \\sin\\left(2\\pi \\frac{{x}}{{{period:.2f}}} + {phase:.2f}\\right) + {B:.2f}"
    print("Fitted Formula in LaTeX format:")
    print(latex_formula)

    return params, period


# 示例用法
prey_data = np.array([2602, 2592, 2305, 2347, 2363, 2489, 2360, 2460, 2363, 2464, 2467, 2386, 2487, 2496, 2499, 2641, 2532, 2642, 2688, 2753, 2686, 2671, 2617, 2542, 2426, 2437, 2421, 2297, 2309, 2229, 2381, 2363, 2565, 2691, 2909, 2669, 2816, 2650, 2616, 2525, 2370, 2481, 2356, 2382, 2388, 2571, 2747, 2675, 2624, 2594, 2553, 2657, 2663, 2631, 2460, 2423, 2334, 2390])
prey_data_11_alive = np.array([2647, 2409, 2420, 2221, 2448, 2313, 2585, 2520, 2669, 2697, 2818, 2696, 2956, 2762, 2849, 2508, 2441, 2376, 2431, 2395, 2574, 2566, 2556, 2503, 2531, 2622, 2499, 2339, 2417, 2278, 2299, 2371, 2435, 2511, 2642, 2576, 2655, 2537, 2661, 2573, 2622, 2437, 2447, 2454, 2558, 2547, 2638, 2671, 2794, 2800, 2767, 2710, 2665, 2560, 2569, 2415, 2464, 2361, 2391])
prey_data_11_birth = np.array([ 3579, 3623, 3526, 3669, 3771, 3682, 3760, 3747, 3824, 3662, 3722, 3608, 3746, 3867, 3762, 3827, 3811, 3793, 3674, 3734, 3769, 3701, 3941, 3801, 4099, 3799, 3905, 3900, 3834, 3857, 3852, 3789, 3785, 3926, 3798, 3752, 3935, 3945, 3791, 3936, 3911, 3697, 3975, 3874, 3932, 3955, 3941, 3969, 3969, 3892, 3853, 3773, 3852, 3676, 3829, 3877, 3786, 3923, 3842])
prey_time = np.arange(1, len(prey_data) + 1)
prey_time_11_alive = np.arange(1, len(prey_data_11_alive) + 1)
prey_time_11_birth = np.arange(1, len(prey_data_11_birth) + 1)


lamprey_data = np.array([ 959, 991, 968, 973, 866, 966, 978, 959, 982, 938, 973, 939, 954, 891, 948, 1016, 994, 1049, 1083, 1033, 1053, 1057, 1010, 914, 964, 868, 909, 813, 930, 924, 1054, 1002, 1135, 1017, 1096, 1089, 1022, 1082, 958, 923, 896, 862, 973, 969, 912, 906, 901, 959, 959, 1086, 1002, 978, 832, 846])
lamprey_data_11_alive = np.array([926, 954, 949, 992, 965, 967, 963, 1037, 1121, 1129, 1169, 1165, 1032, 1018, 948, 969, 1068, 1048, 1102, 1060, 1074, 1103, 1116, 1065, 1134, 1075, 1062, 1012, 1048, 971, 1111, 1016, 1067, 1034, 1074, 1147, 1120, 1066, 1030, 995, 988, 1007, 974, 966, 1055, 1073, 1102, 1075, 1081, 1110, 1105, 1094, 1049, 1021, 1036])
lamprey_data_11_birth = np.array([810, 781, 815, 794, 830, 844, 813, 781, 798, 826, 822, 876, 879, 880, 820, 892, 799, 819, 793, 802, 819, 861, 828, 827, 826, 818, 824, 785, 846, 804, 797, 739, 774, 824, 806, 828, 845, 779, 840, 794, 790, 857, 821, 843, 851, 864, 833, 845, 859, 798, 817, 827, 779, 752, 826])
lamprey_time = np.arange(1, len(lamprey_data) + 1)
lamprey_time_11_alive = np.arange(1, len(lamprey_data_11_alive) + 1)
lamprey_time_11_birth = np.arange(1, len(lamprey_data_11_birth) + 1)



params, period = fit_periodic_function_and_visualize(lamprey_time, lamprey_data)
params, period = fit_periodic_function_and_visualize(lamprey_time_11_alive, lamprey_data_11_alive)
params, period = fit_periodic_function_and_visualize(lamprey_time_11_birth, lamprey_data_11_birth)
