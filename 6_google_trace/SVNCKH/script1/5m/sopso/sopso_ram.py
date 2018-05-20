import sys, os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../../")
from model import sopso
from pandas import read_csv

# pathsave = "/home/thieunv/Desktop/Link to LabThayMinh/code/6_google_trace/SVNCKH/testing/3m/sopso/result/cpu_ram_cpu/"
# fullpath = "/home/thieunv/university/LabThayMinh/code/data/GoogleTrace/"

pathsave = "/home/hunter/nguyenthieu95/ai/6_google_trace/SVNCKH/testing/5m/sopso/result/ram/"
fullpath = "/home/hunter/nguyenthieu95/ai/data/GoogleTrace/"
filename5 = "data_resource_usage_5Minutes_6176858948.csv"
df = read_csv(fullpath+ filename5, header=None, index_col=False, usecols=[4], engine='python')
dataset_original = df.values

list_num5 = (5810, 8300, 1)
output_index = 0
method_statistic = 0
max_cluster=20
neighbourhood_density=0.2
gauss_width=1.0
mutation_id=1

couple_acti = (2, 0)
sliding_windows = [ 2, 3, 5]
positive_numbers =  [0.05, 0.15, 0.35]
stimulation_levels = [0.15, 0.25, 0.45]
distance_levels = [0.55, 0.70, 0.85]

w_minmax = (0.4, 0.9)                               # [0-1] -> [0.4-0.9]      Trong luong cua con chim
value_minmax = (-1, +1)                             # value min of weight
c_couples = [(1.2, 1.2), (0.8, 2.0), (1.6, 0.6)]         #c1, c2 = 2, 2       # [0-2]   Muc do anh huong cua local va global
# r1, r2 : random theo tung vong lap
# delta(t) = 1 (do do: x(sau) = x(truoc) + van_toc
pop_sizes = [80, 150, 250]      # Kich thuoc quan the
max_moves = [90, 150, 400]     # So lan di chuyen`

fig_id = 1
so_vong_lap = 0
for sliding in sliding_windows:
    for pos_number in positive_numbers:
        for sti_level in stimulation_levels:
            for dist_level in distance_levels:

                for max_move in max_moves:
                    for pop_size in pop_sizes:
                        for c_couple in c_couples:

                            my_model = sopso.Model(dataset_original, list_num5, output_index, sliding, method_statistic, max_cluster,
                                             pos_number, sti_level, dist_level, mutation_id, couple_acti, fig_id,pathsave,
                                             max_move, pop_size, c_couple, w_minmax, value_minmax)
                            my_model.fit()
                            so_vong_lap += 1
                            fig_id += 2
                            if so_vong_lap % 100 == 0:
                                print "Vong lap thu : {0}".format(so_vong_lap)
print "Processing DONE !!!"
