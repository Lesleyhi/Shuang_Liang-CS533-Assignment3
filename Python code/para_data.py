# 1000*1000 matrix para test

para_sizes = [32, 64, 96, 128, 160, 192, 224]

# 128 best
Mflops_parasize_blocked = [
    [3923.07, 3896.64, 3913.98, 3920.53, 3923.9, 3892.56, 3908.66, 3909.74, 3899.27, 3898.16, 3913.04, 3896.51,
     3902.67, 3907.39, 3910.78, 3918.42, 3929.28, 3917.5, 3911.42, 3900.99],
    [3855.33, 3851.85, 3850.6, 3846.61, 3855.25, 3849.37, 3845.49, 3855.86, 3856.19, 3858.25, 3846.69, 3855.3,
     3844.43, 3854.78, 3845.8, 3857.38, 3863.57, 3844.23, 3860.71, 3843.91],
    [4024.87, 4022.81, 4024.75, 4021.88, 4024.75, 4024.95, 4025.35, 4024.98, 4024.19, 4023.86, 4024.54, 4024.53,
     4024.24, 4024.31, 4024.3, 4024.47, 4024.07, 4024.67, 4024.36, 4023.76],
    [4195.93, 4195.42, 4195.04, 4196.07, 4196.18, 4195.16, 4195.26, 4196.06, 4196.36, 4195.63, 4195.38, 4196.23,
     4187.7, 4195.13, 4195.55, 4195.41, 4194.59, 4195.05, 4195.84, 4188.07],
    [3694.63, 3684.46, 3694.7, 3685.65, 3694.45, 3694.22, 3692.13, 3693.77, 3694.01, 3694.54, 3694.15, 3694.71,
     3694.04, 3694.49, 3692.44, 3694.81, 3694.47, 3693.85, 3693.04, 3694.36],
    [3761.1, 3751.92, 3761.09, 3751.79, 3760.01, 3759.7, 3752.81, 3761.51, 3761.8, 3761.08, 3760.83, 3760.28,
     3759.79, 3759.38, 3761.22, 3760.48, 3761.08, 3760.49, 3752.23, 3761.75],
    [3798.94, 3792.03, 3797.86, 3789.97, 3800.65, 3799.91, 3789.74, 3798.66, 3799.81, 3799.83, 3799.73, 3798.25,
     3800.35, 3798.36, 3800.3, 3798.07, 3800.47, 3800.07, 3792.75, 3800]
]

# 32
Mflops_parasize_rb = [
    [4194.07, 4194.6, 4194.78, 4194.81, 4193.44, 4194.93, 4195.26, 4193.67, 4193.81, 4194.23, 4194.53, 4195.15,
     4186.15, 4195.4, 4193.02, 4194.65, 4194.61, 4186.9, 4194.49, 4194.49],
    [3998.79, 3991.92, 3998.76, 3991.71, 3997.82, 3996.45, 3993.27, 3998.01, 3996.81, 3997.61, 3998.59, 3990.49,
     3998.38, 3996.35, 3998.23, 3998.11, 3997.41, 3997.05, 3998.17, 3992.7],
    [4094.23, 4088.96, 4094.04, 4087.66, 4093.96, 4093.63, 4087.35, 4094.27, 4094.66, 4093.78, 4093.91, 4093.14,
     4094.6, 4095.06, 4094.43, 4094.87, 4093.71, 4092.49, 4095.27, 4095.06],
    [4055.23, 4055.47, 4059.27, 4056.7, 4057.81, 4057.54, 4059.31, 4058.46, 4059.31, 4057.88, 4058.86, 4058.97,
     4059.12, 4058.19, 4059.35, 4058.47, 4058.84, 4058.43, 4059.4, 4058.85],
    [3654.52, 3647.44, 3654.57, 3645.78, 3655, 3653.54, 3653.39, 3654.21, 3654.32, 3654.72, 3649.3, 3654.68,
     3651.94, 3653.41, 3654.76, 3653.87, 3653.73, 3653.7, 3654.62, 3653.39],
    [3270.63, 3265.44, 3270.58, 3270.53, 3266.37, 3264.78, 3270.96, 3270.99, 3270.34, 3265.14, 3270.87, 3270.14,
     3270.23, 3270.81, 3270.7, 3270.52, 3269.6, 3271.01, 3270.3, 3270.49],
    [3117.81, 3116.61, 3118.34, 3117.41, 3118.38, 3116.24, 3117.75, 3118.06, 3114.01, 3117.65, 3117.7, 3117.21,
     3117.4, 3117.97, 3117.17, 3111.55, 3117.42, 3117.43, 3117.77, 3118.44]
]

# 32
Mflops_parasize_copy = [
    [4232.21, 4233.86, 4233.46, 4233.27, 4233.31, 4233.77, 4233.71, 4233.3, 4233.3, 4233, 4233.83, 4234.31, 4230.51,
     4232.61, 4233.95, 4234.18, 4234.17, 4232.14, 4233.24, 4180.5],
    [3998.7, 3991.85, 3997.92, 3990.73, 3998.57, 3998.39, 3994.87, 3995.46, 3998.32, 3997.58, 3998.36, 3990.02,
     3998.02, 3997.27, 3997.82, 3997.79, 3998.59, 3995.13, 3998.63, 3992.18],
    [3421.86, 3416.12, 3422.14, 3422.16, 3422.28, 3422.11, 3421.19, 3421.84, 3422.15, 3422.04, 3422.71, 3422.25,
     3421.7, 3422.47, 3421.33, 3420.48, 3422.34, 3422.06, 3421.91, 3422.75],
    [3399.55, 3392.81, 3399.79, 3400.12, 3399.18, 3391.31, 3399.97, 3399.33, 3399.13, 3399.61, 3399.75, 3399.81,
     3399.42, 3400.37, 3395.66, 3400.14, 3399.35, 3398.83, 3399.67, 3400.25],
    [3022.72, 3022.92, 3018.68, 3022.78, 3023.61, 3023.27, 3023.27, 3023.61, 3021.18, 3023.28, 3022.77, 3023.06,
     3022.92, 3023.25, 3023.07, 3022.69, 3023.13, 3023.19, 3022.65, 3023.26],
    [2780.06, 2779.94, 2775.09, 2780.17, 2775.38, 2775.12, 2779.31, 2779.88, 2779.07, 2779.52, 2779.94, 2780.01,
     2779.57, 2776.54, 2779.98, 2780.23, 2779.65, 2775.47, 2779.65, 2780.25],
    [2597.53, 2597.4, 2592.88, 2597.62, 2593.36, 2597.45, 2597.67, 2593.53, 2597.52, 2596.17, 2597.02, 2597.3,
     2597.92, 2597.49, 2597.23, 2596.86, 2596.78, 2597.68, 2596.83, 2597.06]
]

# 32
Mflops_parasize_autovect = [
    [4625.16, 4619.99, 4622.45, 4623.8, 4622.93, 4623.84, 4625.64, 4624.87, 4625.26, 4624.51, 4619.96, 4625.22,
     4623.53, 4614.66, 4624.98, 4625.35, 4624.26, 4625.55, 4624.8, 4623.79],
    [4501.82, 4500.81, 4501.14, 4501.43, 4500.37, 4501.08, 4501.86, 4492.03, 4501.65, 4501.62, 4502.32, 4501.07,
     4499.16, 4499.61, 4501.26, 4502, 4502.52, 4502.38, 4501.71, 4502.23],
    [4487.31, 4487.16, 4487.91, 4487.79, 4486.57, 4486.57, 4487.41, 4482.93, 4479.56, 4487.93, 4487.88, 4487.89,
     4487.35, 4487.59, 4487.93, 4486.73, 4487.17, 4487.18, 4487.72, 4486.64],
    [4165.31, 4164.66, 4164.43, 4164.57, 4165.46, 4164.33, 4164.79, 4162.79, 4164.82, 4164.04, 4164.59, 4164.08,
     4161.94, 4163.7, 4164.88, 4164.23, 4164.72, 4162.86, 4165.03, 4164.64],
    [3932.85, 3929.08, 3932.38, 3929.77, 3931.47, 3933.21, 3930, 3931.71, 3931.75, 3931.96, 3932.48, 3929.91,
     3932.68, 3932.32, 3932.8, 3932.93, 3932.22, 3933.7, 3932.7, 3930.01],
    [3590.9, 3583.66, 3591.03, 3589.67, 3591.11, 3590.21, 3590.54, 3591.4, 3592.53, 3592.08, 3584.16, 3591.81,
     3591.29, 3592.39, 3590.97, 3591.89, 3591.51, 3584.63, 3591.81, 3591.47],
    [3393.51, 3390.4, 3394.09, 3394.61, 3393.43, 3391.63, 3393.35, 3393.61, 3393.94, 3393.68, 3393.38, 3393.69,
     3394.15, 3393.88, 3393.1, 3393.93, 3391.83, 3393.8, 3394.23, 3393.4]
]

# 64
Mflops_parasize_genvect = [
    [7090.14, 7083.45, 7055.5, 7087.82, 7088.13, 7082.5, 7064.59, 7027.44, 7087.7, 7079.62, 7092.95, 7090.49,
     7082.01, 7092.93, 7087.65, 7084.26, 7079.14, 7052.51, 7078.42, 7075.61],
    [7905.61, 7905.29, 7905.79, 7892.69, 7902.52, 7905.23, 7904.05, 7901.14, 7903.39, 7906.73, 7904.48, 7903.98,
     7905.38, 7895.99, 7905.98, 7904.68, 7903.07, 7905.17, 7903.83, 7901.93],
    [7646.28, 7647.28, 7647.98, 7616.78, 7647.57, 7645.15, 7645.32, 7646.98, 7646.35, 7644, 7647.16, 7647.13,
     7647.45, 7646.17, 7645.64, 7645.87, 7647.74, 7646.87, 7646.51, 7645.2],
    [5374.61, 5375.12, 5360.81, 5375.21, 5374.19, 5375.75, 5375.71, 5374.9, 5374.34, 5375.1, 5376.01, 5375.46,
     5375.35, 5375.35, 5374.67, 5362.94, 5374.9, 5373.44, 5375.55, 5374.8],
    [5118.57, 5118.57, 5118.68, 5108.7, 5109.73, 5118.15, 5118.18, 5119.06, 5112.58, 5119.73, 5117.77, 5117.49,
     5119.56, 5118.81, 5118.44, 5114.36, 5119.38, 5118.45, 5119.92, 5119.4],
    [4861.57, 4861.25, 4860.12, 4861.79, 4855.81, 4860.84, 4857.25, 4861.72, 4861.37, 4851.24, 4862.65, 4859.38,
     4860.63, 4861.31, 4859.29, 4862.1, 4861.16, 4860.54, 4861.81, 4860.94],
    [4461.61, 4461.35, 4461.17, 4460.96, 4461.95, 4461.94, 4461.11, 4449.77, 4460.27, 4461.37, 4461.23, 4462.27,
     4461.99, 4460.56, 4461.4, 4462.56, 4459.31, 4461.85, 4461.28, 4461]
]
