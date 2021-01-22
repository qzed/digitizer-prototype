project(
    'test',
    'cpp',
    default_options: [
        'cpp_std=c++17',
        'buildtype=debugoptimized',
    ]
)

opt_access_checks = true
opt_fast_math = false

if get_option('fast_math').enabled()
    opt_fast_math = true
elif get_option('fast_math').auto()
    if not get_option('debug') and get_option('optimization') in ['2', '3']
        opt_fast_math = true
    endif
endif

if get_option('access_checks').disabled()
    opt_access_checks = false
elif get_option('fast_math').auto()
    if not get_option('debug') and get_option('optimization') in ['2', '3']
        opt_access_checks = false
    endif
endif

if opt_fast_math
    add_project_arguments('-ffast-math', language: 'cpp')
endif

if opt_access_checks
    add_project_arguments('-DIPTSD_CONFIG_ACCESS_CHECKS', language: 'cpp')
endif

if get_option('cpu_type') != ''
    add_project_arguments('-march=' + get_option('cpu_type'), language: 'cpp')
endif

if get_option('simd_type') != ''
    add_project_arguments('-m' + get_option('simd_type'), language: 'cpp')
endif

cairo_dep = dependency('cairo')

executable('main', 'src/main.cpp', dependencies: [cairo_dep])
