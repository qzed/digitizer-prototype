project(
    'test',
    'cpp', 'c',
    default_options: [
        'cpp_std=c++17',
        'buildtype=debugoptimized',
        'warning_level=3',
    ]
)


# options
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
elif get_option('access_checks').auto()
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

# don't warn about unused parameters for now
add_project_arguments('-Wno-unused-parameter', language: 'cpp')


# dependencies
cairo_dep = dependency('cairo')
gtk_dep = dependency('gtk+-3.0')
threads_dep = dependency('threads')


# binaries
src_plot = [
    'src/proto-plot.cpp',
    'src/processor.cpp',
    'src/visualization.cpp',
]

inc_main = include_directories('src')

src_rt = [
    'src/proto-rt.cpp',
    'src/processor.cpp',
    'src/visualization.cpp',
    'src/control.c',
    'src/utils.c',
]

executable('proto-plot', src_plot, dependencies: [cairo_dep],                       include_directories: inc_main)
executable('proto-rt',   src_rt,   dependencies: [cairo_dep, gtk_dep, threads_dep], include_directories: inc_main)
