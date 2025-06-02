from pymoo.indicators.hv import HV
from pymoo.indicators.gd import GD
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.optimize import minimize
import random
import matplotlib.pyplot as plt
class HistoryCallback:
    def __init__(self):
        self.history = []
    
    def __call__(self, algorithm): 
        # Enregistre la population complète de chaque génération
        self.history.append(algorithm.pop.copy())

def run_optimization(n_jobs, n_stages, plot=True):
    # ======== Configuration des paramètres ========
    print(f"\n{'='*50}")
    print(f"Optimisation pour {n_jobs} jobs et {n_stages} étapes")
    print(f"{'='*50}\n")
    
    # 1. Configuration des ressources selon la structure en 3 étapes
    max_resources = []
    for stage in range(n_stages):
        if stage == 0:  # Étape 1: Collaboration humain qualifié/ordinaire
            max_resources.append([3, 5, 0])  # [Qualifiés, Ordinaires, Robots]
        elif stage == 1:  # Étape 2: Humains ou Robots
            max_resources.append([0, 5, 4])
        else:  # Étapes supplémentaires: Seulement qualifiés
            max_resources.append([4, 0, 0])
    
    # 2. Temps de traitement selon l'article (dictionnaire complet fourni)
    processing_times = {
       (0, 0): (11, 6), (0, 1): (15, 8), (0, 2): (17, 9), (0, 3): (8, 3), (0, 4): (13, 7),
        (1, 0): (8, 2), (1, 1): (6, 1), (1, 2): (7, 2), (1, 3): (9, 1), (1, 4): (5, 2),
        (2, 0): (13, 2), (2, 1): (10, 1), (2, 2): (9, 2), (2, 3): (7, 1), (2, 4): (11, 2),
        (3, 0): (7, 3), (3, 1): (12, 7), (3, 2): (9, 5), (3, 3): (14, 8), (3, 4): (6, 4),
        (4, 0): (14, 1), (4, 1): (8, 2), (4, 2): (11, 1), (4, 3): (9, 2), (4, 4): (12, 1),
        (5, 0): (9, 4), (5, 1): (11, 7), (5, 2): (13, 8), (5, 3): (6, 3), (5, 4): (10, 6),
        (6, 0): (10, 1), (6, 1): (7, 2), (6, 2): (12, 1), (6, 3): (8, 2), (6, 4): (9, 1),
        (7, 0): (12, 5), (7, 1): (14, 8), (7, 2): (11, 6), (7, 3): (13, 7), (7, 4): (7, 3),
        (8, 0): (6, 2), (8, 1): (9, 1), (8, 2): (10, 2), (8, 3): (5, 1), (8, 4): (8, 2),
        (9, 0): (15, 7), (9, 1): (10, 5), (9, 2): (8, 3), (9, 3): (11, 6), (9, 4): (14, 9),
        (10, 0): (7, 1), (10, 1): (11, 2), (10, 2): (6, 1), (10, 3): (10, 2), (10, 4): (9, 1),
        (11, 0): (10, 5), (11, 1): (13, 8), (11, 2): (7, 3), (11, 3): (12, 7), (11, 4): (9, 4),
        (12, 0): (8, 2), (12, 1): (12, 1), (12, 2): (9, 2), (12, 3): (6, 1), (12, 4): (11, 2),
        (13, 0): (13, 6), (13, 1): (9, 4), (13, 2): (11, 7), (13, 3): (15, 9), (13, 4): (7, 3),
        (14, 0): (9, 1), (14, 1): (10, 2), (14, 2): (13, 1), (14, 3): (7, 2), (14, 4): (12, 1),
        (15, 0): (11, 6), (15, 1): (8, 3), (15, 2): (10, 5), (15, 3): (14, 8), (15, 4): (6, 2),
        (16, 0): (7, 2), (16, 1): (13, 1), (16, 2): (8, 2), (16, 3): (11, 1), (16, 4): (9, 2),
        (17, 0): (14, 7), (17, 1): (9, 4), (17, 2): (12, 6), (17, 3): (7, 3), (17, 4): (10, 5),
        (18, 0): (10, 1), (18, 1): (6, 2), (18, 2): (11, 1), (18, 3): (13, 2), (18, 4): (8, 1),
        (19, 0): (12, 6), (19, 1): (7, 3), (19, 2): (14, 8), (19, 3): (10, 5), (19, 4): (9, 4),
        (20, 0): (8, 1), (20, 1): (12, 3), (20, 2): (14, 2), (20, 3): (7, 1), (20, 4): (9, 3),
        (21, 0): (14, 7), (21, 1): (10, 5), (21, 2): (7, 2), (21, 3): (12, 8), (21, 4): (9, 4),
        (22, 0): (6, 2), (22, 1): (10, 1), (22, 2): (13, 3), (22, 3): (8, 1), (22, 4): (11, 2),
        (23, 0): (12, 7), (23, 1): (8, 4), (23, 2): (11, 6), (23, 3): (15, 9), (23, 4): (7, 2),
        (24, 0): (9, 1), (24, 1): (15, 3), (24, 2): (11, 2), (24, 3): (6, 1), (24, 4): (13, 3),
        (25, 0): (7, 3), (25, 1): (11, 6), (25, 2): (14, 9), (25, 3): (9, 4), (25, 4): (13, 7),
        (26, 0): (10, 2), (26, 1): (14, 1), (26, 2): (8, 3), (26, 3): (12, 1), (26, 4): (7, 2),
        (27, 0): (16, 8), (27, 1): (9, 4), (27, 2): (7, 2), (27, 3): (11, 6), (27, 4): (13, 7),
        (28, 0): (11, 1), (28, 1): (13, 3), (28, 2): (9, 2), (28, 3): (15, 1), (28, 4): (8, 3),
        (29, 0): (8, 4), (29, 1): (10, 6), (29, 2): (12, 8), (29, 3): (6, 3), (29, 4): (14, 9),
        (30, 0): (12, 6), (30, 1): (7, 3), (30, 2): (14, 8), (30, 3): (10, 5), (30, 4): (9, 4),
        (31, 0): (8, 2), (31, 1): (6, 1), (31, 2): (7, 2), (31, 3): (9, 1), (31, 4): (5, 2),
        (32, 0): (13, 2), (32, 1): (10, 1), (32, 2): (9, 2), (32, 3): (7, 1), (32, 4): (11, 2),
        (33, 0): (7, 3), (33, 1): (12, 7), (33, 2): (9, 5), (33, 3): (14, 8), (33, 4): (6, 4),
        (34, 0): (14, 1), (34, 1): (8, 2), (34, 2): (11, 1), (34, 3): (9, 2), (34, 4): (12, 1),
        (35, 0): (9, 4), (35, 1): (11, 7), (35, 2): (13, 8), (35, 3): (6, 3), (35, 4): (10, 6),
        (36, 0): (10, 1), (36, 1): (7, 2), (36, 2): (12, 1), (36, 3): (8, 2), (36, 4): (9, 1),
        (37, 0): (12, 5), (37, 1): (14, 8), (37, 2): (11, 6), (37, 3): (13, 7), (37, 4): (7, 3),
        (38, 0): (6, 2), (38, 1): (9, 1), (38, 2): (10, 2), (38, 3): (5, 1), (38, 4): (8, 2),
        (39, 0): (15, 7), (39, 1): (10, 5), (39, 2): (8, 3), (39, 3): (11, 6), (39, 4): (14, 9),
        (40, 0): (7, 1), (40, 1): (11, 2), (40, 2): (6, 1), (40, 3): (10, 2), (40, 4): (9, 1),
        (41, 0): (10, 5), (41, 1): (13, 8), (41, 2): (7, 3), (41, 3): (12, 7), (41, 4): (9, 4),
        (42, 0): (8, 2), (42, 1): (12, 1), (42, 2): (9, 2), (42, 3): (6, 1), (42, 4): (11, 2),
        (43, 0): (13, 6), (43, 1): (9, 4), (43, 2): (11, 7), (43, 3): (15, 9), (43, 4): (7, 3),
        (44, 0): (9, 1), (44, 1): (10, 2), (44, 2): (13, 1), (44, 3): (7, 2), (44, 4): (12, 1),
        (45, 0): (11, 6), (45, 1): (8, 3), (45, 2): (10, 5), (45, 3): (14, 8), (45, 4): (6, 2),
        (46, 0): (7, 2), (46, 1): (13, 1), (46, 2): (8, 2), (46, 3): (11, 1), (46, 4): (9, 2),
        (47, 0): (14, 7), (47, 1): (9, 4), (47, 2): (12, 6), (47, 3): (7, 3), (47, 4): (10, 5),
        (48, 0): (10, 1), (48, 1): (6, 2), (48, 2): (11, 1), (48, 3): (13, 2), (48, 4): (8, 1),
        (49, 0): (12, 6), (49, 1): (7, 3), (49, 2): (14, 8), (49, 3): (10, 5), (49, 4): (9, 4),
        (50, 0): (8, 1), (50, 1): (12, 3), (50, 2): (14, 2), (50, 3): (7, 1), (50, 4): (9, 3),
        (51, 0): (14, 7), (51, 1): (10, 5), (51, 2): (7, 2), (51, 3): (12, 8), (51, 4): (9, 4),
        (52, 0): (6, 2), (52, 1): (10, 1), (52, 2): (13, 3), (52, 3): (8, 1), (52, 4): (11, 2),
        (53, 0): (12, 7), (53, 1): (8, 4), (53, 2): (11, 6), (53, 3): (15, 9), (53, 4): (7, 2),
        (54, 0): (9, 1), (54, 1): (15, 3), (54, 2): (11, 2), (54, 3): (6, 1), (54, 4): (13, 3),
        (55, 0): (7, 3), (55, 1): (11, 6), (55, 2): (14, 9), (55, 3): (9, 4), (55, 4): (13, 7),
        (56, 0): (10, 2), (56, 1): (14, 1), (56, 2): (8, 3), (56, 3): (12, 1), (56, 4): (7, 2),
        (57, 0): (16, 8), (57, 1): (9, 4), (57, 2): (7, 2), (57, 3): (11, 6), (57, 4): (13, 7),
        (58, 0): (11, 1), (58, 1): (13, 3), (58, 2): (9, 2), (58, 3): (15, 1), (58, 4): (8, 3),
        (59, 0): (8, 4), (59, 1): (10, 6), (59, 2): (12, 8), (59, 3): (6, 3), (59, 4): (14, 9),
        (60, 0): (12, 6), (60, 1): (7, 3), (60, 2): (14, 8), (60, 3): (10, 5), (60, 4): (9, 4),
        (61, 0): (8, 2), (61, 1): (6, 1), (61, 2): (7, 2), (61, 3): (9, 1), (61, 4): (5, 2),
        (62, 0): (13, 2), (62, 1): (10, 1), (62, 2): (9, 2), (62, 3): (7, 1), (62, 4): (11, 2),
        (63, 0): (7, 3), (63, 1): (12, 7), (63, 2): (9, 5), (63, 3): (14, 8), (63, 4): (6, 4),
        (64, 0): (14, 1), (64, 1): (8, 2), (64, 2): (11, 1), (64, 3): (9, 2), (64, 4): (12, 1),
        (65, 0): (9, 4), (65, 1): (11, 7), (65, 2): (13, 8), (65, 3): (6, 3), (65, 4): (10, 6),
        (66, 0): (10, 1), (66, 1): (7, 2), (66, 2): (12, 1), (66, 3): (8, 2), (66, 4): (9, 1),
        (67, 0): (12, 5), (67, 1): (14, 8), (67, 2): (11, 6), (67, 3): (13, 7), (67, 4): (7, 3),
        (68, 0): (6, 2), (68, 1): (9, 1), (68, 2): (10, 2), (68, 3): (5, 1), (68, 4): (8, 2),
        (69, 0): (15, 7), (69, 1): (10, 5), (69, 2): (8, 3), (69, 3): (11, 6), (69, 4): (14, 9),
        (70, 0): (7, 1), (70, 1): (11, 2), (70, 2): (6, 1), (70, 3): (10, 2), (70, 4): (9, 1),
        (71, 0): (10, 5), (71, 1): (13, 8), (71, 2): (7, 3), (71, 3): (12, 7), (71, 4): (9, 4),
        (72, 0): (8, 2), (72, 1): (12, 1), (72, 2): (9, 2), (72, 3): (6, 1), (72, 4): (11, 2),
        (73, 0): (13, 6), (73, 1): (9, 4), (73, 2): (11, 7), (73, 3): (15, 9), (73, 4): (7, 3),
        (74, 0): (9, 1), (74, 1): (10, 2), (74, 2): (13, 1), (74, 3): (7, 2), (74, 4): (12, 1),
        (75, 0): (11, 6), (75, 1): (8, 3), (75, 2): (10, 5), (75, 3): (14, 8), (75, 4): (6, 2),
        (76, 0): (7, 2), (76, 1): (13, 1), (76, 2): (8, 2), (76, 3): (11, 1), (76, 4): (9, 2),
        (77, 0): (14, 7), (77, 1): (9, 4), (77, 2): (12, 6), (77, 3): (7, 3), (77, 4): (10, 5),
        (78, 0): (10, 1), (78, 1): (6, 2), (78, 2): (11, 1), (78, 3): (13, 2), (78, 4): (8, 1),
        (79, 0): (12, 6), (79, 1): (7, 3), (79, 2): (14, 8), (79, 3): (10, 5), (79, 4): (9, 4),
        (80, 0): (8, 1), (80, 1): (12, 3), (80, 2): (14, 2), (80, 3): (7, 1), (80, 4): (9, 3),
        (81, 0): (14, 7), (81, 1): (10, 5), (81, 2): (7, 2), (81, 3): (12, 8), (81, 4): (9, 4),
        (82, 0): (6, 2), (82, 1): (10, 1), (82, 2): (13, 3), (82, 3): (8, 1), (82, 4): (11, 2),
        (83, 0): (12, 7), (83, 1): (8, 4), (83, 2): (11, 6), (83, 3): (15, 9), (83, 4): (7, 2),
        (84, 0): (9, 1), (84, 1): (15, 3), (84, 2): (11, 2), (84, 3): (6, 1), (84, 4): (13, 3),
        (85, 0): (7, 3), (85, 1): (11, 6), (85, 2): (14, 9), (85, 3): (9, 4), (85, 4): (13, 7),
        (86, 0): (10, 2), (86, 1): (14, 1), (86, 2): (8, 3), (86, 3): (12, 1), (86, 4): (7, 2),
        (87, 0): (16, 8), (87, 1): (9, 4), (87, 2): (7, 2), (87, 3): (11, 6), (87, 4): (13, 7),
        (88, 0): (11, 1), (88, 1): (13, 3), (88, 2): (9, 2), (88, 3): (15, 1), (88, 4): (8, 3),
        (89, 0): (8, 4), (89, 1): (10, 6), (89, 2): (12, 8), (89, 3): (6, 3), (89, 4): (14, 9),
        (90, 0): (12, 6), (90, 1): (7, 3), (90, 2): (14, 8), (90, 3): (10, 5), (90, 4): (9, 4),
        (91, 0): (8, 2), (91, 1): (6, 1), (91, 2): (7, 2), (91, 3): (9, 1), (91, 4): (5, 2),
        (92, 0): (13, 2), (92, 1): (10, 1), (92, 2): (9, 2), (92, 3): (7, 1), (92, 4): (11, 2),
        (93, 0): (7, 3), (93, 1): (12, 7), (93, 2): (9, 5), (93, 3): (14, 8), (93, 4): (6, 4),
        (94, 0): (14, 1), (94, 1): (8, 2), (94, 2): (11, 1), (94, 3): (9, 2), (94, 4): (12, 1),
        (95, 0): (9, 4), (95, 1): (11, 7), (95, 2): (13, 8), (95, 3): (6, 3), (95, 4): (10, 6),
        (96, 0): (10, 1), (96, 1): (7, 2), (96, 2): (12, 1), (96, 3): (8, 2), (96, 4): (9, 1),
        (97, 0): (12, 5), (97, 1): (14, 8), (97, 2): (11, 6), (97, 3): (13, 7), (97, 4): (7, 3),
        (98, 0): (6, 2), (98, 1): (9, 1), (98, 2): (10, 2), (98, 3): (5, 1), (98, 4): (8, 2),
        (99, 0): (15, 7), (99, 1): (10, 5), (99, 2): (8, 3), (99, 3): (11, 6), (99, 4): (14, 9),
        (100, 0): (7, 1), (100, 1): (11, 2), (100, 2): (6, 1), (100, 3): (10, 2), (100, 4): (9, 1),
        (101, 0): (10, 5), (101, 1): (13, 8), (101, 2): (7, 3), (101, 3): (12, 7), (101, 4): (9, 4),
        (102, 0): (8, 2), (102, 1): (12, 1), (102, 2): (9, 2), (102, 3): (6, 1), (102, 4): (11, 2),
        (103, 0): (13, 6), (103, 1): (9, 4), (103, 2): (11, 7), (103, 3): (15, 9), (103, 4): (7, 3),
        (104, 0): (9, 1), (104, 1): (10, 2), (104, 2): (13, 1), (104, 3): (7, 2), (104, 4): (12, 1),
        (105, 0): (11, 6), (105, 1): (8, 3), (105, 2): (10, 5), (105, 3): (14, 8), (105, 4): (6, 2),
        (106, 0): (7, 2), (106, 1): (13, 1), (106, 2): (8, 2), (106, 3): (11, 1), (106, 4): (9, 2),
        (107, 0): (14, 7), (107, 1): (9, 4), (107, 2): (12, 6), (107, 3): (7, 3), (107, 4): (10, 5),
        (108, 0): (10, 1), (108, 1): (6, 2), (108, 2): (11, 1), (108, 3): (13, 2), (108, 4): (8, 1),
        (109, 0): (12, 6), (109, 1): (7, 3), (109, 2): (14, 8), (109, 3): (10, 5), (109, 4): (9, 4),
        (110, 0): (8, 1), (110, 1): (12, 3), (110, 2): (14, 2), (110, 3): (7, 1), (110, 4): (9, 3),
        (111, 0): (14, 7), (111, 1): (10, 5), (111, 2): (7, 2), (111, 3): (12, 8), (111, 4): (9, 4),
        (112, 0): (6, 2), (112, 1): (10, 1), (112, 2): (13, 3), (112, 3): (8, 1), (112, 4): (11, 2),
        (113, 0): (12, 7), (113, 1): (8, 4), (113, 2): (11, 6), (113, 3): (15, 9), (113, 4): (7, 2),
        (114, 0): (9, 1), (114, 1): (15, 3), (114, 2): (11, 2), (114, 3): (6, 1), (114, 4): (13, 3),
        (115, 0): (7, 3), (115, 1): (11, 6), (115, 2): (14, 9), (115, 3): (9, 4), (115, 4): (13, 7),
        (116, 0): (10, 2), (116, 1): (14, 1), (116, 2): (8, 3), (116, 3): (12, 1), (116, 4): (7, 2),
        (117, 0): (16, 8), (117, 1): (9, 4), (117, 2): (7, 2), (117, 3): (11, 6), (117, 4): (13, 7),
        (118, 0): (11, 1), (118, 1): (13, 3), (118, 2): (9, 2), (118, 3): (15, 1), (118, 4): (8, 3),
        (119, 0): (8, 4), (119, 1): (10, 6), (119, 2): (12, 8), (119, 3): (6, 3), (119, 4): (14, 9),
        (120, 0): (12, 6), (120, 1): (7, 3), (120, 2): (14, 8), (120, 3): (10, 5), (120, 4): (9, 4),
        (121, 0): (8, 2), (121, 1): (6, 1), (121, 2): (7, 2), (121, 3): (9, 1), (121, 4): (5, 2),
        (122, 0): (13, 2), (122, 1): (10, 1), (122, 2): (9, 2), (122, 3): (7, 1), (122, 4): (11, 2),
        (123, 0): (7, 3), (123, 1): (12, 7), (123, 2): (9, 5), (123, 3): (14, 8), (123, 4): (6, 4),
        (124, 0): (14, 1), (124, 1): (8, 2), (124, 2): (11, 1), (124, 3): (9, 2), (124, 4): (12, 1),
        (125, 0): (9, 4), (125, 1): (11, 7), (125, 2): (13, 8), (125, 3): (6, 3), (125, 4): (10, 6),
        (126, 0): (10, 1), (126, 1): (7, 2), (126, 2): (12, 1), (126, 3): (8, 2), (126, 4): (9, 1),
        (127, 0): (12, 5), (127, 1): (14, 8), (127, 2): (11, 6), (127, 3): (13, 7), (127, 4): (7, 3),
        (128, 0): (6, 2), (128, 1): (9, 1), (128, 2): (10, 2), (128, 3): (5, 1), (128, 4): (8, 2),
        (129, 0): (15, 7), (129, 1): (10, 5), (129, 2): (8, 3), (129, 3): (11, 6), (129, 4): (14, 9),
        (130, 0): (7, 1), (130, 1): (11, 2), (130, 2): (6, 1), (130, 3): (10, 2), (130, 4): (9, 1),
        (131, 0): (10, 5), (131, 1): (13, 8), (131, 2): (7, 3), (131, 3): (12, 7), (131, 4): (9, 4),
        (132, 0): (8, 2), (132, 1): (12, 1), (132, 2): (9, 2), (132, 3): (6, 1), (132, 4): (11, 2),
        (133, 0): (13, 6), (133, 1): (9, 4), (133, 2): (11, 7), (133, 3): (15, 9), (133, 4): (7, 3),
        (134, 0): (9, 1), (134, 1): (10, 2), (134, 2): (13, 1), (134, 3): (7, 2), (134, 4): (12, 1),
        (135, 0): (11, 6), (135, 1): (8, 3), (135, 2): (10, 5), (135, 3): (14, 8), (135, 4): (6, 2),
        (136, 0): (7, 2), (136, 1): (13, 1), (136, 2): (8, 2), (136, 3): (11, 1), (136, 4): (9, 2),
        (137, 0): (14, 7), (137, 1): (9, 4), (137, 2): (12, 6), (137, 3): (7, 3), (137, 4): (10, 5),
        (138, 0): (10, 1), (138, 1): (6, 2), (138, 2): (11, 1), (138, 3): (13, 2), (138, 4): (8, 1),
        (139, 0): (12, 6), (139, 1): (7, 3), (139, 2): (14, 8), (139, 3): (10, 5), (139, 4): (9, 4)
    }
    
    # 3. Paramètres énergétiques
    P_run = [1.0] * n_stages  # kW par ressource active
    P_sb  = [0.5] * n_stages  # kW en mode veille

    # ======== Définition du problème ========
    class HFSProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(
                n_var=n_jobs,
                n_obj=2,
                n_constr=0,
                xl=0,
                xu=n_jobs-1,
                vtype=int
            )
            self.max_resources = max_resources
            self.processing_times = processing_times
            self.P_run = P_run
            self.P_sb = P_sb
        
        def assign_resources(self, job, stage):
            """Assignation des ressources selon les règles de l'étape"""
            max_skilled, max_ordinary, max_robots = self.max_resources[stage]
            
            # Étape 1: Collaboration qualifié + ordinaire
            if stage == 0:
                N_s = random.randint(1, max_skilled)
                N_o = random.randint(0, max_ordinary)
                return N_s, N_o, 0
            
            # Étape 2: Choix entre ordinaires ou robots
            elif stage == 1:
                if random.random() < 0.5:
                    N_o = random.randint(1, max_ordinary)
                    return 0, N_o, 0
                else:
                    R = random.randint(1, max_robots)
                    return 0, 0, R
            
            # Autres étapes: Seulement qualifiés
            else:
                N_s = random.randint(1, max_skilled)
                return N_s, 0, 0
        
        def calculate_completion_times(self, permutation):
            """Calcul des temps d'achèvement selon les Éq. 6-9 de l'article"""
            completion_times = np.zeros((n_stages, n_jobs))
            resource_usage = []
            
            for stage in range(n_stages):
                stage_times = []
                
                for pos, job_id in enumerate(permutation):
                    job = job_id
                    
                    # Assignation des ressources
                    N_s, N_o, R = self.assign_resources(job, stage)
                    total_workers = N_s + N_o + R
                    
                    # Récupération du temps de traitement standard
                    # (Gestion des étapes au-delà de 4 avec recyclage)
                    stage_key = stage if stage < 5 else stage % 5
                    p_time = self.processing_times.get((job, stage_key), (10, 0))[0]
                    
                    # Temps de traitement réel (Éq. 5)
                    duration = p_time / total_workers
                    
                    # Calcul du temps de début (Éq. 6-9)
                    if stage == 0 and pos == 0:
                        start_time = 0
                    elif stage == 0:
                        start_time = completion_times[stage][pos-1]
                    elif pos == 0:
                        start_time = completion_times[stage-1][pos]
                    else:
                        start_time = max(
                            completion_times[stage][pos-1],
                            completion_times[stage-1][pos]
                        )
                    
                    # Mise à jour des temps
                    end_time = start_time + duration
                    completion_times[stage][pos] = end_time
                    stage_times.append((start_time, end_time, job, N_s, N_o, R))
                
                resource_usage.append(stage_times)
            
            return completion_times, resource_usage
        
        def calculate_energy(self, completion_times, resource_usage):
            """Calcul de la consommation énergétique selon les Éq. 3-4 de l'article"""
            EC_r = 0.0  # Énergie active
            EC_s = 0.0  # Énergie en veille
            
            # Calcul pour chaque étape
            for stage in range(n_stages):
                stage_times = resource_usage[stage]
                total_processing = 0.0
                
                # Calcul de l'énergie active (Éq. 3)
                for job_data in stage_times:
                    start, end, job_id, N_s, N_o, R = job_data
                    duration = end - start
                    total_workers = N_s + N_o + R
                    EC_r += total_workers * duration * self.P_run[stage]
                    total_processing += duration
                
                # Calcul de l'énergie en veille (Éq. 4)
                stage_makespan = np.max(completion_times[stage])
                standby_time = stage_makespan - total_processing
                EC_s += standby_time * self.P_sb[stage]
            
            return EC_r, EC_s
        
        def _evaluate(self, x, out, *args, **kwargs):
            permutation = x.astype(int)
            completion_times, resource_usage = self.calculate_completion_times(permutation)
            
            # Makespan (Éq. 1)
            makespan = np.max(completion_times[-1])
            
            # TEC (Éq. 2)
            EC_r, EC_s = self.calculate_energy(completion_times, resource_usage)
            TEC = EC_r + EC_s
            
            out["F"] = [makespan, TEC]

    # ======== Configuration de l'algorithme ========
    algorithm = SPEA2(
        pop_size=min(100, n_jobs*2),
        sampling=PermutationRandomSampling(),
        crossover=OrderCrossover(),
        mutation=InversionMutation(),
        eliminate_duplicates=True
    )

    # ======== Optimisation ========
    problem = HFSProblem()
    res = minimize(
        problem,
        algorithm,
        ('n_gen', max(50, n_jobs//2)),
        seed=random.randint(1, 1000),
        verbose=False
    )
# Initialisation du callback pour l'historique
    history_callback = HistoryCallback() 
    
    res = minimize(
        problem,
        algorithm,
        ('n_gen', max(50, n_jobs//2)),
        seed=random.randint(1, 1000),
        verbose=False,
        callback=history_callback
    )
    # ======== Résultats (syntaxe préservée) ========
    print(f"Results for {n_jobs} jobs and {n_stages} stages:")
    
    if len(res.pop) > 0:
        best_makespan = min(res.pop, key=lambda ind: ind.F[0])
        best_energy = min(res.pop, key=lambda ind: ind.F[1])
        
        print("\nBest makespan solution:")
        print(f"  Makespan = {best_makespan.F[0]:.2f}")
        print(f"  TEC = {best_makespan.F[1]:.2f}")
        
        print("\nBest energy solution:")
        print(f"  Makespan = {best_energy.F[0]:.2f}")
        print(f"  TEC = {best_energy.F[1]:.2f}")
        
        # Visualisation conditionnelle
        if plot:
              plot_results(res, n_jobs, n_stages, history_callback.history)
        print("No solutions found!")
    
    return res, history_callback.history
     

def plot_results(res, n_jobs, n_stages, history):
    """Fonction de visualisation avec calcul réel de HV et GD"""
    if res is None or not hasattr(res, 'pop') or len(res.pop) == 0:
        print(f"No results to plot for {n_jobs} jobs and {n_stages} stages")
        return
    
    plt.figure(figsize=(15, 4))
    
    # 1. Front de Pareto
    plt.subplot(131)
    F = np.array([ind.F for ind in res.pop])
    plt.scatter(F[:, 1], F[:, 0], c='blue', alpha=0.7)
    plt.xlabel('Total Energy Consumption (TEC)')
    plt.ylabel('Makespan')
    plt.title(f'Pareto Front ({n_jobs} jobs, {n_stages} stages)')
    plt.grid(True)
    
    # 2. Calcul des indicateurs HV et GD
    if history:
        # Préparation des données
        gen_count = len(history)
        hv_vals = []
        gd_vals = []
        
        # Création du point de référence pour HV (10% au-dessus du maximum observé)
        all_objs = np.array([ind.F for gen in history for ind in gen])
        ref_point = np.max(all_objs, axis=0) * 1.10
        
        # Création du front de référence pour GD (front final)
        pf = np.array([ind.F for ind in res.opt]) if hasattr(res, 'opt') else np.array([])
        
        # Création des instances d'indicateurs
        hv_calculator = HV(ref_point=ref_point)
        gd_calculator = GD(pf)
        
        # Calcul des valeurs pour chaque génération
        for gen in history:
            objs = np.array([ind.F for ind in gen])
            hv_vals.append(hv_calculator(objs))
            gd_vals.append(gd_calculator(objs))
        
        # 2. Évolution de l'Hypervolume
        plt.subplot(132)
        plt.plot(range(1, gen_count+1), hv_vals, 'g-', linewidth=2)
        plt.xlabel('Generation')
        plt.ylabel('HV Value')
        plt.title('Hypervolume (HV) Evolution')
        plt.grid(True)
        
        # 3. Évolution de la Distance Générationnelle
        plt.subplot(133)
        plt.plot(range(1, gen_count+1), gd_vals, 'r-', linewidth=2)
        plt.xlabel('Generation')
        plt.ylabel('GD Value')
        plt.title('Generational Distance (GD) Evolution')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'results_{n_jobs}j_{n_stages}s.png', dpi=300)
    plt.show()
    # ======== Exécution principale ========
if __name__ == "__main__":
    # Configuration des instances à tester
    configs = [
        (20, 3),   # Petite instance
        (60, 5),   # Instance moyenne
        (140, 10)  # Grande instance
    ]
    
    for n_jobs, n_stages in configs:
        print(f"\n{'='*50}")
        print(f"Running optimization for {n_jobs} jobs and {n_stages} stages...")
        res = run_optimization(n_jobs, n_stages, plot=True)
        
        print(f"\n{'='*50}")
        print(f"Completed optimization for {n_jobs} jobs and {n_stages} stages")
        print(f"{'='*50}\n")