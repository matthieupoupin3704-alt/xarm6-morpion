import cv2
import cv2.aruco as aruco
import numpy as np
import time
import json
import os
import math
import copy
import random


# --- CONSTANTES EXTERNES (Au cas où si fichier non trouvé) ---
try:
    from constants import MINIMAX_INF, MINIMAX_VICTOIRE
except ImportError:
    MINIMAX_INF = 1000      # Valeur "infinie" pour l'initialisation minimax (score impossible en pratique)
    MINIMAX_VICTOIRE = 10   # Score attribué à une victoire . Doit être > nombre de cases (9) pour rester dominant

# --- ROBOT IP CONFIGURATION ---
IP_ROBOT = '192.168.1.208'  # Adresse IP fixe du bras xArm sur le réseau local . À modifier si le réseau change

# --- DIMENSIONS PHYSIQUES ---
# Ces constantes définissent la géométrie réelle du setup. Une erreur ici se traduit
# directement par un décalage vertical de la pince (pion raté ou plateau percuté).
TAILLE_PION = 50.0        # Hauteur totale du pion en mm (cube 5×5×5 cm) . Non utilisé directement dans les calculs Z
Z_PRISE_OFFSET = 15.0     # Distance (mm) entre la BASE du pion et le point de fermeture de la pince
                           # = moitié de TAILLE_PION pour saisir au milieu du pion
                           # DEBUG: si la pince ripe (trop bas) ou saisit le dessus (trop haut), ajuster ici
EPAISSEUR_PLATEAU = 10.0  # Épaisseur du plateau de jeu en mm . Utilisé pour calculer z_sol = h_jeu - EPAISSEUR_PLATEAU
                           # DEBUG: si le robot enfonce les pions dans le plateau, cette valeur est trop grande
ACTIVER_HAUTEUR_LACHER_VARIABLE = False  # True: ajuste z_drop_offset selon voisins. False: hauteur de lacher fixe.
ACTIVER_DETECTION_TRICHE = False          # False: desactive toute la detection d'incoherence plateau (triche)

# --- PARAMETRES DE PERFORMANCE ROBOT ---
# Augmentation moderee de vitesse/acceleration pour reduire la duree d'un tour sans degrader la stabilite.
ROBOT_SPEED_SCALE = 1.25
ROBOT_ACC_SCALE = 1.20
ROBOT_MAX_SPEED = 900
ROBOT_MAX_ACC = 2200
WAIT_GRIPPER_CHECK_S = 0.30
WAIT_STOCK_ANALYSE_S = 0.10
WAIT_PLATEAU_ANALYSE_S = 0.20
WAIT_POST_DROP_S = 0.12
WAIT_AJUSTEMENT_VISEE_S = 0.16

# --- CONSTANTES COULEURS HUD ---
# Note: OpenCV utilise BGR (pas RGB). C_CYAN=(255,255,0) = jaune en BGR pur, mais rendu cyan sur fond sombre.
C_CYAN   = (255, 255, 0)    # Couleur du robot (O) . Utilisée pour les cercles, la barre de statut robot, les textes de mouvement
C_ORANGE = (0, 165, 255)    # Couleur de l'humain (X) . Utilisée pour les croix et la barre de statut "ATTENTE HUMAIN"
C_GRIS   = (100, 100, 100)  # Couleur des cases vides (coins) et de la grille de la minimap
C_TENUE  = (200, 200, 200)  # Couleur neutre (match nul, sous-titres de fin de partie)
C_VERT   = (0, 255, 0)      # Couleur de confirmation visuelle (marqueur ArUco centré, label YAW dans _centrer_camera)
C_ROUGE  = (0, 0, 255)      # Couleur d'alerte (barre de validation coup humain, erreurs)

# ============================================================================
# 0. GESTION DE LA CONFIGURATION (JSON) ET UTILITAIRES
# ============================================================================

FICHIER_CONFIG = "config_robot.json"  # Chemin relatif du fichier de calibration . Créé dans le répertoire courant d'exécution

# Table de translittération caractères accentués → ASCII
# Nécessaire car cv2.putText() ne gère pas l'Unicode . Les accents produisent des carrés ou crashent
_ACCENT_MAP = str.maketrans(
    "aàâäAÀÂeéèêëEÉÈÊiîïIÎoôöOÔuùûüUÙÛcçCÇnñ—–×·",
    "aaaaAAAeeeeeEEEEiiiIIoooOOuuuuUUUccCCnn--x."
)

def _cv2txt(s: str) -> str:
    """Convertit une chaine en ASCII pur pour cv2.putText."""
    return str(s).translate(_ACCENT_MAP)

CONFIG_DEFAULT = {
    # offset_cam: vecteur [dx, dy, dz] en mm entre le TCP du robot et l'objectif de la caméra
    # DEBUG: si la caméra pointe à côté de la cible, ajuster offset_cam[0] (X) et [1] (Y)
    "offset_cam": [-50.0, 0.0, 115.0],
    # correction_mecanique_xy: biais résiduel après calibration (mm), corrige le drift physique
    # DEBUG: si le robot se trompe toujours du même côté, c'est cette valeur qui est fausse
    "correction_mecanique_xy": [0.0, -15.0],
    # pos_scan: [X, Y, Z, roll, pitch, yaw] . Position de survol pour la vue globale du plateau
    "pos_scan": [300.0, 0.0, 300.0, 180.0, 0.0, 0.0],
    # pos_stock: position au-dessus de la réserve de pions du robot
    "pos_stock": [200.0, 250.0, 15.0, 180.0, 0.0, 0.0],
    "hauteurs": {
        "vol": 100.0,   # Hauteur de transit entre les points (mm) . Assez haute pour éviter les obstacles
        "jeu": 25.0     # Hauteur de la surface du plateau (mm) . Calibrée manuellement
    },
    "vision": {
        "gamma": 2.0,       # Correction gamma appliquée avant détection ArUco (>1 = éclaircit)
        "clahe_clip": 4.0   # Limite du CLAHE pour le rehaussement de contraste local
    },
    "params_mvt": {
        "speed_fast": 700,  # Vitesse rapide (mm/s) pour les déplacements de transit
        "speed_slow": 150,  # Vitesse lente (mm/s) pour la descente sur le pion (précision)
        "acc": 1600        # Accélération (mm/s²)
    }
}

def _update_dict_recursif(d, u):
    # Fusionne u dans d en profondeur . Les clés de u écrasent celles de d
    # Permet de charger un fichier de config partiel sans perdre les valeurs par défaut
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = _update_dict_recursif(d.get(k, {}), v)
        else:
            d[k] = v
    return d

class ConfigManager:
    def __init__(self):
        self.fichier_existe = False  # True si config_robot.json a été trouvé et chargé avec succès
                                     # Utilisé dans main() pour décider si la calibration est obligatoire
        self.data = copy.deepcopy(CONFIG_DEFAULT)  # dict de configuration actif . Deepcopy pour ne pas modifier CONFIG_DEFAULT
        self.charger()  # Écrase self.data avec les valeurs du fichier JSON si disponible

    def charger(self):
        if os.path.exists(FICHIER_CONFIG):
            self.fichier_existe = True
            try:
                with open(FICHIER_CONFIG, 'r') as f:
                    saved = json.load(f)
                    # Merge récursif : les valeurs du fichier priment sur les défauts
                    self.data = _update_dict_recursif(self.data, saved)
                print(f"Configuration chargee depuis {FICHIER_CONFIG}.")
            except Exception as e:
                print(f"Erreur lecture configuration : {e}")
        else:
            self.fichier_existe = False
            print("Fichier de configuration non detecte. Une calibration OBLIGATOIRE sera requise.")

    def sauvegarder(self):
        try:
            with open(FICHIER_CONFIG, 'w') as f:
                json.dump(self.data, f, indent=4)
            print(f"Configuration sauvegardee dans {FICHIER_CONFIG}.")
        except Exception as e:
            print(f"Erreur sauvegarde configuration : {e}")

cfg = ConfigManager()  # Instance globale unique . Accédée partout via cfg.data["clé"]

# --- Parametres ArUco ---
# Chaque objet physique porte un marqueur ArUco imprimé avec un ID unique.
# La plage d'IDs sépare les rôles : 0-3=coins, 10-18=cases, 20-29=robot, 30-39=humain.

ID_COINS_PLATEAU = [0, 1, 2, 3]
# ID_COINS_PLATEAU : marqueurs collés aux 4 coins physiques du plateau.
# Actuellement non utilisés dans les calculs (réservés pour une future homographie).

MAPPING_CASES = {10:0, 11:1, 12:2, 13:3, 14:4, 15:5, 16:6, 17:7, 18:8}
# MAPPING_CASES : dict ArUco_ID → index_case (0 à 8).
# Index des cases : 0=haut-gauche, 1=haut-centre, ..., 8=bas-droite (lecture gauche→droite, haut→bas).
# Chaque case du plateau porte un marqueur ArUco (ID 10-18) sur la surface du plateau.
# Quand un pion est posé dessus, le marqueur disparaît . C'est le signal de détection de coup, en plus de la détection des pions eux-mêmes (IDs 20-39).
# DEBUG: si un coup est joué sur la mauvaise case, vérifier ce mapping ET l'orientation physique des marqueurs.
CASE_TO_ARUCO = {case_idx: aruco_id for aruco_id, case_idx in MAPPING_CASES.items()}

ID_PIONS_ROBOT  = list(range(20, 30))  # IDs 20-29 : marqueurs collés sur les pions du robot (O)
ID_PIONS_HUMAIN = list(range(30, 40))  # IDs 30-39 : marqueurs collés sur les pions de l'humain (ou de l'adversaire)(X)
# Note: seuls les IDs effectivement présents sur le matériel sont détectés .
# les IDs 20-29 non utilisés physiquement n'apparaîtront jamais dans ids_flat.

# --- DETECTION MATERIEL ---
# Drapeaux booléens fixés au démarrage selon les imports disponibles.
# Si False, le code bascule en mode simulation sans avoir besoin de try/except dans chaque méthode.
try:
    from xarm.wrapper import XArmAPI
    XARM_AVAILABLE = True   # True = librairie xArm installée → connexion réelle possible
except ImportError:
    XARM_AVAILABLE = False  # False = pas de librairie → mode simulation forcé

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True   # True = SDK RealSense installé → caméra réelle possible
except ImportError:
    REALSENSE_AVAILABLE = False  # False = pas de SDK → image noire simulée

# ============================================================================
# 1. CERVEAU DU JEU #OK
# ============================================================================

# Niveaux de difficulté du robot 
#   FACILE    : quasi 70% aléatoire, saisit quand même les victoires immédiates
#   MOYEN     : gagne > bloque > centre > coin > aléatoire (pas de planification)
#   DIFFICILE : minimax alpha-beta
DIFFICULTES: tuple[str, ...] = ("FACILE", "MOYEN", "DIFFICILE")
_PROBA_RANDOM_FACILE = 0.7  # Part d'aléatoire en mode FACILE, on peut l'ajuster

Plateau = list[str]  # Alias de type : liste de 9 caractères, chaque élément vaut "X", "O" ou " " (vide)

# 8 combinaisons gagnantes : 3 lignes, 3 colonnes, 2 diagonales
# Chaque sous-liste contient les 3 indices de cases qui forment une ligne gagnante.
# Utilisé par verifier_gagnant() pour tester toutes les conditions de victoire en une passe.
_LIGNES_GAGNANTES: list[list[int]] = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8],   # lignes horizontales (haut, milieu, bas)
    [0, 3, 6], [1, 4, 7], [2, 5, 8],   # colonnes verticales (gauche, centre, droite)
    [0, 4, 8], [2, 4, 6],              # diagonales
]

# Ordre de visite des cases pour l'algorithme minimax et meilleur_coup().
# Trier les cases dans cet ordre avant d'explorer améliore l'élagage alpha-beta :
# les meilleurs coups (centre, coins) sont évalués en premier → plus de coupures.
# Centre(4) > coins(0,2,6,8) > bords(1,3,5,7)
_ORDRE_STRATEGIQUE: list[int] = [4, 0, 2, 6, 8, 1, 3, 5, 7]


def verifier_gagnant(plateau: Plateau, joueur: str) -> bool:
    # Retourne True si le joueur occupe une ligne gagnante complète
    return any(all(plateau[i] == joueur for i in ligne) for ligne in _LIGNES_GAGNANTES)

def est_nul(plateau: Plateau) -> bool:
    # Nul si plus aucune case libre (et aucun gagnant . À vérifier en amont)
    return " " not in plateau

def coup_gagnant(plateau: Plateau, joueur: str) -> int:
    """Retourne l'index du coup gagnant immédiat pour le joueur, ou -1 s'il n'existe pas.
    Utilisé à la fois pour attaquer (O) et pour bloquer (X) dans meilleur_coup."""
    for i in range(9):
        if plateau[i] != " ":
            continue
        plateau[i] = joueur
        gagne = verifier_gagnant(plateau, joueur)
        plateau[i] = " "  # Annule le coup test . Le plateau n'est PAS modifié ici
        if gagne:
            return i
    return -1

class MorpionIA:
    def __init__(self, difficulte: str = "DIFFICILE") -> None:
        self.difficulte = difficulte if difficulte in DIFFICULTES else "DIFFICILE"

    def set_difficulte(self, niveau: str) -> None:
        """Change le niveau de jeu . Validé contre DIFFICULTES pour éviter les typos."""
        if niveau in DIFFICULTES:
            self.difficulte = niveau

    def meilleur_coup(self, plateau: Plateau) -> int:
        """Retourne l'index de la case à jouer pour le robot (O).
        Stratégie selon self.difficulte :
          - FACILE    : ~70% random, saisit quand même les victoires évidentes
          - MOYEN     : gagne > bloque > centre > random (pas de planification)
          - DIFFICILE : IA parfaite (minimax alpha-beta)"""
        cases_libres = [i for i, v in enumerate(plateau) if v == " "]
        if not cases_libres:
            return -1  # Plateau plein . Ne devrait pas arriver si est_nul() est testé avant

        if self.difficulte == "FACILE":
            return self._coup_facile(plateau, cases_libres)
        if self.difficulte == "MOYEN":
            return self._coup_moyen(plateau, cases_libres)
        return self._coup_difficile(plateau, cases_libres)

    def _coup_facile(self, plateau: Plateau, cases_libres: list[int]) -> int:
        if random.random() >= _PROBA_RANDOM_FACILE:
            coup = coup_gagnant(plateau, "O")
            if coup != -1:
                return coup
        return random.choice(cases_libres)

    def _coup_moyen(self, plateau: Plateau, cases_libres: list[int]) -> int:
        # Joue les coups "évidents" mais ne planifie pas : l'humain peut créer un
        # double-menace (fork) . C'est exactement la marge laissée volontairement.
        coup = coup_gagnant(plateau, "O")
        if coup != -1: return coup
        coup = coup_gagnant(plateau, "X")
        if coup != -1: return coup
        if plateau[4] == " ":
            return 4
        coins = [c for c in (0, 2, 6, 8) if c in cases_libres]
        if coins:
            return random.choice(coins)
        return random.choice(cases_libres)

    def _coup_difficile(self, plateau: Plateau, cases_libres: list[int]) -> int:
        # Shortcuts avant minimax (évite un calcul inutile sur les cas triviaux)
        coup = coup_gagnant(plateau, "O")
        if coup != -1: return coup
        coup = coup_gagnant(plateau, "X")
        if coup != -1: return coup
        if len(cases_libres) >= 8 and plateau[4] == " ":
            return 4

        # Minimax alpha-beta, cases triées pour maximiser l'élagage
        cases_triees = sorted(cases_libres, key=lambda x: _ORDRE_STRATEGIQUE.index(x) if x in _ORDRE_STRATEGIQUE else 9)
        meilleure_valeur = -MINIMAX_INF
        meilleur_indice: int = cases_triees[0]

        for i in cases_triees:
            plateau[i] = "O"
            valeur = self._minimax(plateau, profondeur=0, maximise=False, alpha=-MINIMAX_INF, beta=MINIMAX_INF)
            plateau[i] = " "
            if valeur > meilleure_valeur:
                meilleure_valeur = valeur
                meilleur_indice = i

        return meilleur_indice

    def _minimax(self, plateau: Plateau, profondeur: int, maximise: bool, alpha: int, beta: int) -> int:
        """
        Paramètres :
          profondeur : nombre de coups simulés depuis la racine . Pénalise les victoires tardives
          maximise   : True = tour du robot (cherche le max), False = tour humain (cherche le min)
          alpha      : meilleur score garanti pour le robot dans la branche courante (borne basse)
          beta       : meilleur score garanti pour l'humain dans la branche courante (borne haute)
                       Si beta <= alpha -> coupure : l'autre joueur ne choisira jamais cette branche
        """
        # Cas terminaux : score positif si O gagne (robot), négatif si X gagne (humain)
        # La profondeur pénalise les victoires tardives (favorise les coups rapides)
        if verifier_gagnant(plateau, "O"): return MINIMAX_VICTOIRE - profondeur
        if verifier_gagnant(plateau, "X"): return -(MINIMAX_VICTOIRE - profondeur)
        if est_nul(plateau): return 0

        # Tri par ordre stratégique pour améliorer l'élagage alpha-beta (meilleurs coups d'abord)
        cases_libres = sorted([i for i, v in enumerate(plateau) if v == " "], key=lambda x: _ORDRE_STRATEGIQUE.index(x) if x in _ORDRE_STRATEGIQUE else 9)

        if maximise:
            # Tour du robot (O) : on cherche le max
            meilleur = -MINIMAX_INF  # Valeur courante maximale trouvée dans cette branche
            for i in cases_libres:
                plateau[i] = "O"
                valeur = self._minimax(plateau, profondeur + 1, False, alpha, beta)
                plateau[i] = " "
                meilleur = max(meilleur, valeur)
                alpha = max(alpha, meilleur)  # Met à jour la borne basse du robot
                if beta <= alpha: break       # Coupure beta : l'adversaire ne choisira jamais cette branche
            return meilleur
        else:
            # Tour de l'humain (X) : on cherche le min
            meilleur = MINIMAX_INF  # Valeur courante minimale trouvée dans cette branche
            for i in cases_libres:
                plateau[i] = "X"
                valeur = self._minimax(plateau, profondeur + 1, True, alpha, beta)
                plateau[i] = " "
                meilleur = min(meilleur, valeur)
                beta = min(beta, meilleur)    # Met à jour la borne haute de l'humain
                if beta <= alpha: break       # Coupure alpha : le robot ne choisira jamais cette branche
            return meilleur


def choisir_difficulte_terminal(defaut: str = "DIFFICILE") -> str:
    """Demande le niveau de l'IA en CLI. Entree vide = valeur par defaut.
    Accepte le numero (1/2/3) ou le nom (facile/moyen/difficile), insensible a la casse."""
    print("\n[MAIN] Choix du niveau de l'IA :")
    for i, nom in enumerate(DIFFICULTES, start=1):
        marque = " (defaut)" if nom == defaut else ""
        print(f"  {i}. {nom}{marque}")
    while True:
        reponse = input("[MAIN] Votre choix [1-3] : ").strip().upper()
        if not reponse:
            return defaut
        if reponse.isdigit() and 1 <= int(reponse) <= len(DIFFICULTES):
            return DIFFICULTES[int(reponse) - 1]
        if reponse in DIFFICULTES:
            return reponse
        print("[MAIN] Entree invalide. Tapez 1, 2 ou 3 (ou Entree pour le defaut).")


# ============================================================================
# 2. GESTION DU ROBOT (REEL OU SIMULE)
# ============================================================================

class RobotController:
    def __init__(self, ip, config): #OK
        self.ip = ip        # Adresse IP du robot (string) . Transmise à XArmAPI pour la connexion TCP
        self.cfg = config   # Référence au dict cfg.data . Contient toutes les positions et paramètres de mouvement
        self.arm = None     # Instance XArmAPI . None tant que non connecté ou en mode simulation
        self.last_sequence_alert = ""  # Derniere alerte operationnelle (affichable dans le HUD)
        self.last_drop_info = None  # Memoire du dernier depot (case, ArUco cible, XY)
        # simule=True si xArm non installé OU si la connexion échoue
        # Toutes les méthodes vérifient self.simule avant d'appeler self.arm
        self.simule = not XARM_AVAILABLE
        self.connect()
        # Après connect() : self.dummy_pos peut être absent (créé au 1er deplacer() en simulation)

    def connect(self): #OK
        if not self.simule:
            try:
                self.arm = XArmAPI(self.ip, is_single_thread_cb=True)
                self.arm.clean_error()
                self.arm.clean_warn()
                self.arm.motion_enable(enable=True)
                self.arm.set_mode(0)   # Mode position cartésienne
                self.arm.set_state(state=0)  # État : prêt
                self.arm.set_gripper_enable(True)  # Active la pince
                self.arm.set_gripper_mode(0) # Mode de contrôle de la pince : position
                time.sleep(1) #Le temps que la pince s'initialise
                self.gripper(ouvrir=False)
                print(f"Robot connecté avec succès sur {self.ip}. Pince fermée.")
            except Exception as e:
                print(f"Connexion échouée ({e}) -> PASSAGE EN MODE SIMULATION")
                self.simule = True
        else:
            print("Mode Simulation Force (Librairie xArm non installée)")

    def get_pose(self): #OK
        """Retourne la pose actuelle [X, Y, Z, roll, pitch, yaw] en mm/degrés.
        En simulation, renvoie dummy_pos (dernière cible envoyée) ou pos_scan si aucun mouvement fait."""
        if self.simule:
            if hasattr(self, 'dummy_pos'): return self.dummy_pos
            return self.cfg["pos_scan"]

        code, pos = self.arm.get_position()
        if code == 0: return pos
        # DEBUG: code != 0 → erreur SDK xArm, vérifier l'état du robot avec arm.get_state()
        return [0,0,0,0,0,0]

    def sleep_visuel(self, duration, vision, win_name, message=""):
        """Attente active qui continue d'afficher le flux caméra pendant la pause.
        Évite que la fenêtre freeze . Indispensable sur les longs déplacements."""
        t0 = time.time()
        while time.time() - t0 < duration:
            if vision and win_name:
                img, _, _ = vision.capturer()
                if img is not None:
                    if message:
                        cv2.rectangle(img, (0, 0), (640, 40), (20, 20, 20), -1)
                        cv2.putText(img, _cv2txt(message), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_CYAN, 2, cv2.LINE_AA)
                    cv2.imshow(win_name, img)
                cv2.waitKey(10)
            else:
                time.sleep(0.05)

    def deplacer(self, x, y, z, roll=180, pitch=0, yaw=0, speed=None, wait=True, vision=None, win_name=None, message=""):
        """Déplace le TCP vers (x, y, z) en mm dans le repère robot.
        roll=180 oriente la pince vers le bas (convention xArm).

        DEBUG: si le robot ne bouge pas et qu'on est en mode réel, vérifier :
          - arm.get_state() → doit être 0 (prêt) ou 2 (arrêt)
          - arm.get_err_warn_code() pour les erreurs matérielles
        """
        base_speed = speed if speed else self.cfg["params_mvt"]["speed_fast"]
        base_acc = self.cfg["params_mvt"]["acc"]
        s = int(min(ROBOT_MAX_SPEED, max(50, round(base_speed * ROBOT_SPEED_SCALE))))
        acc = int(min(ROBOT_MAX_ACC, max(200, round(base_acc * ROBOT_ACC_SCALE))))

        if not self.simule and self.arm:
            for tentative in range(3):  # 3 tentatives max avant abandon
                try:
                    if wait and vision and win_name:
                        # Déplacement non-bloquant + polling de l'état pour garder l'UI vivante
                        code = self.arm.set_position(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw, speed=s, mvacc=acc, wait=False)
                        if code == 0:
                            time.sleep(0.1)  # Laisse le robot démarrer le mouvement avant de poller
                            while True:
                                _, state = self.arm.get_state()
                                # state=1 = en mouvement, state=2 = arrêt, state=4 = pause
                                if state != 1: break
                                img, _, _ = vision.capturer()
                                if img is not None:
                                    if message:
                                        cv2.rectangle(img, (0, 0), (640, 40), (20, 20, 20), -1)
                                        cv2.putText(img, _cv2txt(message), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_CYAN, 2, cv2.LINE_AA)
                                    cv2.imshow(win_name, img)
                                cv2.waitKey(10)
                            break
                        else:
                            # code != 0 : erreur xArm . Purge et réessaie
                            self.recuperer_erreur_robot()
                    else:
                        code = self.arm.set_position(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw, speed=s, mvacc=acc, wait=wait)
                        if code == 0: break
                        else: self.recuperer_erreur_robot()
                except Exception as e:
                    self.recuperer_erreur_robot()
        else:
            # Mode simulation : on simule la durée du déplacement avec l'UI active
            if wait and vision and win_name:
                self.sleep_visuel(0.4, vision, win_name, message)
            elif wait:
                time.sleep(0.05)
            # dummy_pos mémorise la dernière cible pour que get_pose() renvoie quelque chose de cohérent
            self.dummy_pos = [x, y, z, roll, pitch, yaw]

    def recuperer_erreur_robot(self): #OK
        """Purge les erreurs matérielles xArm et remet le robot en état opérationnel.
        Appelé automatiquement après chaque code d'erreur . Ne pas ignorer les erreurs répétées."""
        if self.simule or not self.arm: return
        print("[ROBOT] Purge des erreurs matérielles en cours...")
        try:
            self.arm.clean_error()
            self.arm.clean_warn()
            self.arm.clean_gripper_error()
            self.arm.motion_enable(enable=True)
            self.arm.set_state(state=0)
            time.sleep(0.5)
        except Exception as e:
            pass

    def retour_scan_securise(self, vision=None, win_name=None): #OK
        """Retour vers la position de scan en passant d'abord à la hauteur de sécurité.
        Évite les collisions si le robot est en position basse au-dessus du plateau."""
        scan = self.cfg["pos_scan"]
        h_scan = scan[2]  # Hauteur de la position scan (Z en mm)
        v_fast = self.cfg["params_mvt"]["speed_fast"]

        pos_actuelle = self.get_pose()
        # Remontée préventive si on est en dessous de la hauteur de scan
        if pos_actuelle[2] < h_scan:
            self.deplacer(pos_actuelle[0], pos_actuelle[1], h_scan, roll=pos_actuelle[3], pitch=pos_actuelle[4], yaw=pos_actuelle[5], speed=v_fast, vision=vision, win_name=win_name, message="RETOUR HAUTEUR SECURITE")

        self.deplacer(scan[0], scan[1], scan[2], roll=scan[3], pitch=scan[4], yaw=scan[5], speed=v_fast, vision=vision, win_name=win_name, message="RETOUR POSITION SCAN")

    def gripper(self, ouvrir=True, check_grab=False, force_drop=False, vision=None, win_name=None): #OK
        """Contrôle la pince. Position 850 = ouvert, 0 = fermé.

        check_grab=True : vérifie que la pince a bien saisi quelque chose (pos > 50 après fermeture)
        force_drop=True : vérifie que la pince s'est bien ouverte (pos > 600) et secoue si non

        DEBUG: si current_pos < 50 après fermeture → pince vide (a raté le pion)
               si current_pos < 600 après ouverture → pion coincé dans la pince
        """
        if not self.simule and self.arm:
            pos = 850 if ouvrir else 0
            for tentative in range(3):
                try:
                    code = self.arm.set_gripper_position(pos, wait=True)
                    if code != 0:
                        self.recuperer_erreur_robot()
                        continue

                    if check_grab and not ouvrir:
                        # Vérifie que la pince tient bien quelque chose après fermeture
                        self.sleep_visuel(WAIT_GRIPPER_CHECK_S, vision, win_name, "VERIFICATION PRISE PION")
                        code_pos, current_pos = self.arm.get_gripper_position()
                        # Si la pince est quasi-fermée (< 50), elle a fermé dans le vide
                        if code_pos == 0 and current_pos < 50: return False
                        return True

                    if force_drop and ouvrir:
                        # Vérifie que le pion est bien lâché après ouverture
                        self.sleep_visuel(WAIT_GRIPPER_CHECK_S, vision, win_name, "VERIFICATION LACHER PION")
                        code_pos, current_pos = self.arm.get_gripper_position()
                        # Si la pince n'est pas assez ouverte, le pion est peut-être encore dedans
                        return True

                    return True
                except Exception as e:
                    self.recuperer_erreur_robot()
            return False
        else:
            return True

    def danse_victoire(self, vision=None, win_name=None):
        """Animation de victoire revue : montee, vague laterale, clap pince, retour neutre."""
        if self.simule: return

        v_appro = 220
        v_dance = 320
        v_snap = 420
        scan = self.cfg["pos_scan"]
        x, y, z, r, _, yw = scan

        self.retour_scan_securise(vision, win_name)
        self.gripper(ouvrir=True, vision=vision, win_name=win_name)

        # 1) Montee celebratoire
        z_haut = z + 28
        self.deplacer(x, y, z_haut, roll=r, pitch=10, yaw=yw,
                      speed=v_appro, vision=vision, win_name=win_name, message="VICTOIRE !")
        self.sleep_visuel(0.12, vision, win_name, "VICTOIRE !")

        # 2) Vague laterale (mouvements courts, meme envelope securite)
        amp_x = 22
        seq = [
            (x + amp_x, z_haut + 4, yw + 26, 12),
            (x - amp_x, z_haut + 2, yw - 26, 12),
            (x + amp_x, z_haut + 4, yw + 22, 8),
            (x - amp_x, z_haut + 2, yw - 22, 8),
        ]
        for x_t, z_t, yaw_t, pitch_t in seq:
            self.deplacer(x_t, y, z_t, roll=r, pitch=pitch_t, yaw=yaw_t,
                          speed=v_dance, vision=vision, win_name=win_name, message="VICTOIRE !")

        # 3) Clap de pince rapide pour finir la celebration
        self.deplacer(x, y, z_haut + 6, roll=r, pitch=6, yaw=yw,
                      speed=v_dance, vision=vision, win_name=win_name, message="VICTOIRE !")
        for _ in range(2):
            self.gripper(ouvrir=False, vision=vision, win_name=win_name)
            self.sleep_visuel(0.10, vision, win_name, "VICTOIRE !")
            self.gripper(ouvrir=True, vision=vision, win_name=win_name)
            self.sleep_visuel(0.10, vision, win_name, "VICTOIRE !")

        # 4) Petit rebond final puis retour propre a la pose scan
        self.deplacer(x, y, z + 4, roll=r, pitch=-8, yaw=yw,
                      speed=v_snap, vision=vision, win_name=win_name, message="VICTOIRE !")
        self.deplacer(x, y, z + 18, roll=r, pitch=6, yaw=yw,
                      speed=v_snap, vision=vision, win_name=win_name, message="VICTOIRE !")
        self.deplacer(x, y, z, roll=r, pitch=0, yaw=yw,
                      speed=v_appro, vision=vision, win_name=win_name)
        self.gripper(ouvrir=False, vision=vision, win_name=win_name)

    def animation_nul(self, vision=None, win_name=None):
        """Poignée de main : la pince se tend vers l'humain, s'incline en ouvrant/fermant comme
        si elle serrait une main, puis oscille verticalement pour simuler le mouvement de
        'shake'. L'inclinaison du poignet (pitch) donne l'illusion d'un vrai serrage."""
        if self.simule: return
        v_appro = 200   # Vitesse d'approche vers la position de serrage
        v_shake = 350   # Vitesse des oscillations du shake (rapide = naturel)
        scan = self.cfg["pos_scan"]
        x, y, z, r, _, yw = scan

        self.retour_scan_securise(vision, win_name)

        # 1. Tendre la pince vers l'humain : pince en avant (x+60mm), pitch a 0
        #    pour que la pince pointe horizontalement, gripper ouvert.
        self.gripper(ouvrir=True, vision=vision, win_name=win_name)
        x_hand = x + 80
        z_hand = z + 10  # Un peu plus haut que scan pour etre a hauteur de main
        self.deplacer(x_hand, y, z_hand, roll=r, pitch=0, yaw=yw,
                      speed=v_appro, vision=vision, win_name=win_name, message="BIEN JOUE !")

        # 2. Fermer la pince ("serre la main")
        self.gripper(ouvrir=False, vision=vision, win_name=win_name)
        self.sleep_visuel(0.2, vision, win_name, "BIEN JOUE !")

        # 3. Mouvement de "shake" : oscillation verticale courte avec inclinaison du poignet.
        #    L'inclinaison alternee du pitch (+/- 15deg) imite le flechissement naturel du
        #    poignet humain. Amplitude Z reduite (~15mm) pour que le geste soit sec et net.
        for _ in range(3):
            self.deplacer(x_hand, y, z_hand + 15, roll=r, pitch=+15, yaw=yw,
                          speed=v_shake, vision=vision, win_name=win_name, message="BIEN JOUE !")
            self.deplacer(x_hand, y, z_hand - 15, roll=r, pitch=-15, yaw=yw,
                          speed=v_shake, vision=vision, win_name=win_name, message="BIEN JOUE !")

        # 4. Relacher et retour neutre
        self.deplacer(x_hand, y, z_hand, roll=r, pitch=0, yaw=yw,
                      speed=v_appro, vision=vision, win_name=win_name, message="BIEN JOUE !")
        self.gripper(ouvrir=True, vision=vision, win_name=win_name)
        self.retour_scan_securise(vision, win_name)

    def animation_defaite(self, vision=None, win_name=None):
        """Animation de defaite : 'tete baissee' (pitch negatif prononce), petite secousse
        en signe de deception, puis releve lent. Plus expressive que la version precedente
        qui se contentait d'une seule inclinaison statique."""
        if self.simule: return
        v_lent = 80
        v_sec = 180
        scan = self.cfg["pos_scan"]
        x, y, z, r, _, yw = scan

        self.retour_scan_securise(vision, win_name)

        # 1. Tete baissee lentement : pitch -55deg, Z descend, epaules "tombees"
        x_def = x - 30
        z_def = z - 30
        self.deplacer(x_def, y, z_def, roll=r, pitch=-55, yaw=yw,
                      speed=v_lent, vision=vision, win_name=win_name, message="DECEPTION...")
        self.sleep_visuel(0.6, vision, win_name, "DECEPTION...")

        # 2. Petite secousse laterale : le robot "hoche la tete" de gauche a droite
        #    en signe de denegation. Yaw oscille de +/- 20 deg autour de yw.
        for _ in range(2):
            self.deplacer(x_def, y, z_def, roll=r, pitch=-55, yaw=yw + 20,
                          speed=v_sec, vision=vision, win_name=win_name, message="DECEPTION...")
            self.deplacer(x_def, y, z_def, roll=r, pitch=-55, yaw=yw - 20,
                          speed=v_sec, vision=vision, win_name=win_name, message="DECEPTION...")
        self.deplacer(x_def, y, z_def, roll=r, pitch=-55, yaw=yw,
                      speed=v_sec, vision=vision, win_name=win_name, message="DECEPTION...")
        self.sleep_visuel(0.4, vision, win_name, "DECEPTION...")

        # 3. Releve lent vers la pose scan
        self.retour_scan_securise(vision, win_name)

    def evaluer_rotation_pince_stock(self, target_center, target_corners, tous_centres): #OK
        """Calcule le yaw optimal de la pince pour saisir un pion dans le stock.

        Paramètres :
          target_center  : (cx, cy) pixel du centre du pion cible
          target_corners : array (4, 2) des coins du marqueur ArUco du pion cible
          tous_centres   : liste de (cx, cy) de tous les pions robot visibles dans le stock
                           (utilisée pour détecter les voisins proches)

        Logique :
          1. Aligne la pince sur l'axe du marqueur ArUco (angle du bord c0→c1)
          2. Si un voisin est proche (<120px) et horizontal → pivote de 90° pour éviter la collision

        DEBUG: si le robot tord systématiquement les pions, imprimer angle_cube_deg et yaw_optimal ici.
        """
        c0, c1, c2, c3 = target_corners
        # Vecteur du bord c0→c1 du marqueur = axe principal du cube
        dx_edge = c1[0] - c0[0]  # Composante horizontale du vecteur bord
        dy_edge = c1[1] - c0[1]  # Composante verticale du vecteur bord

        # Angle dans l'image, ramené dans [-45°, +45°] (symétrie du carré : 4 orientations identiques)
        angle_cube_deg = -math.degrees(math.atan2(dy_edge, dx_edge))
        yaw_base = angle_cube_deg % 90
        if yaw_base > 45: yaw_base -= 90
        elif yaw_base < -45: yaw_base += 90

        # Recherche du voisin le plus proche pour détecter une contrainte d'espace
        min_dist = float('inf')   # Distance au voisin le plus proche (pixels)
        closest_neighbor = None   # Coordonnées (cx, cy) du voisin le plus proche

        for center in tous_centres:
            if center == target_center: continue
            dist = math.hypot(center[0] - target_center[0], center[1] - target_center[1])
            if dist < min_dist:
                min_dist = dist
                closest_neighbor = center

        yaw_optimal = yaw_base
        if closest_neighbor and min_dist < 120:  # 120px ≈ proximité gênante
            dx_voisin = abs(closest_neighbor[0] - target_center[0])  # Écart horizontal au voisin
            dy_voisin = abs(closest_neighbor[1] - target_center[1])  # Écart vertical au voisin
            # Si le voisin est plus à droite/gauche que haut/bas → voisinage horizontal → tourne la pince
            if dx_voisin > dy_voisin:
                yaw_optimal += 90

        # Normalisation dans [-90°, +90°] . La pince est symétrique, +/-180° = même position
        while yaw_optimal > 90: yaw_optimal -= 180
        while yaw_optimal < -90: yaw_optimal += 180

        return round(yaw_optimal)  # Retourne un entier (degrés) passé directement à deplacer(..., yaw=...)

    def evaluer_rotation_pince_plateau(self, cible_idx, board_ia, target_corners=None): #OK
        """Calcule le yaw de la pince ET l'offset Z de dépôt pour une case du plateau.

        Paramètres :
          cible_idx      : index de case cible (0-8)
          board_ia       : état logique actuel du plateau . Utilisé pour détecter les voisins occupés
          target_corners : coins ArUco de la case cible (optionnel) . Si fourni, aligne sur le marqueur

        Retourne (yaw_optimal, z_drop_offset) :
          - yaw_optimal    : angle (degrés) pour aligner la pince dans l'espace libre de la case
          - z_drop_offset  : hauteur supplémentaire de lâcher si des pions voisins gênent (mm)
            → 0mm si case isolée, 10mm si 1-2 voisins, 25mm si très entouré

        DEBUG: si le pion tombe à côté, vérifier z_drop_offset (trop haut = trop de dérive à la chute).
        """
        yaw_base = 0  # Angle de base de la pince . Mis à jour si target_corners est fourni
        if target_corners is not None:
            c0, c1, c2, c3 = target_corners
            dx_edge = c1[0] - c0[0]
            dy_edge = c1[1] - c0[1]
            angle_case_deg = -math.degrees(math.atan2(dy_edge, dx_edge))

            # Ramène dans [-45°, +45°]
            yaw_base = angle_case_deg % 90
            if yaw_base > 45: yaw_base -= 90
            elif yaw_base < -45: yaw_base += 90

        yaw_optimal = yaw_base  # Angle final de la pince . Potentiellement pivoté de 90° selon les voisins
        # Construction des listes de voisins selon la ligne (3x3 grid)
        voisins_h = []  # Indices des cases directement à gauche et à droite de cible_idx
        voisins_v = []  # Indices des cases directement au-dessus et en-dessous de cible_idx

        if cible_idx in [0, 1, 2]:  # Ligne haute
            if cible_idx > 0: voisins_h.append(cible_idx - 1)
            if cible_idx < 2: voisins_h.append(cible_idx + 1)
            voisins_v.append(cible_idx + 3)
        elif cible_idx in [3, 4, 5]:  # Ligne centrale
            if cible_idx > 3: voisins_h.append(cible_idx - 1)
            if cible_idx < 5: voisins_h.append(cible_idx + 1)
            voisins_v.append(cible_idx - 3)
            voisins_v.append(cible_idx + 3)
        elif cible_idx in [6, 7, 8]:  # Ligne basse
            if cible_idx > 6: voisins_h.append(cible_idx - 1)
            if cible_idx < 8: voisins_h.append(cible_idx + 1)
            voisins_v.append(cible_idx - 3)

        pions_v_presents = any(board_ia[v] != " " for v in voisins_v)  # True si au moins un voisin vertical est occupé
        pions_h_presents = any(board_ia[v] != " " for v in voisins_h)  # True si au moins un voisin horizontal est occupé

        nb_neighbors = sum(1 for v in voisins_h + voisins_v if board_ia[v] != " ")  # Nombre total de voisins occupés (0 à 4)

        # [MODIF FIABILITE DEPOT 2026-04-20] Lâcher progressif et homogène :
        # - Avant : 0 voisin -> 0 mm (risque de riper au sol), 1-2 voisins -> 10 mm, 3+ -> 25 mm
        # - Après : paliers plus doux et toujours un petit offset minimum pour ne pas racler le plateau
        #   0 voisin : 2 mm (plancher minimum . Évite le "ras" qui bloque le pion)
        #   1 voisin : 6 mm
        #   2 voisins : 12 mm
        #   3+ voisins OU croix V+H : 20 mm
        if nb_neighbors >= 3 or (pions_v_presents and pions_h_presents):
            z_drop_offset = 20.0
        elif nb_neighbors == 2:
            z_drop_offset = 12.0
        elif nb_neighbors == 1:
            z_drop_offset = 6.0
        else:
            z_drop_offset = 2.0  # Case isolée : mini-offset pour un lâcher net

        if not ACTIVER_HAUTEUR_LACHER_VARIABLE:
            z_drop_offset = 0.0

        # Si des pions sont en V mais pas en H → orienter la pince horizontalement pour passer
        if pions_v_presents and not pions_h_presents:
            yaw_optimal += 90

        while yaw_optimal > 90: yaw_optimal -= 180
        while yaw_optimal < -90: yaw_optimal += 180

        return round(yaw_optimal), z_drop_offset

    def _centrer_camera_sur_cible(self, vision, valid_ids, z_target, h_scan, win_name, is_stock=False, ids_exclus=None, board_ia=None, cible_idx=None):
        """Affine la position du robot pour centrer la caméra sur la cible ArUco.

        [MODIF FIABILITE 2026-04-20] Plusieurs améliorations de robustesse :
          1. Correction mécanique (correction_mecanique_xy) appliquée DANS les itérations
             -> la caméra vise la vraie cible physique, pas une cible décalée par le biais.
          2. Verrouillage de l'ID cible dès la 1ère itération (mode stock)
             -> évite de changer de pion à mi-convergence.
          3. Sortie anticipée si le dernier déplacement commandé est infinitésimal (hystérésis).
          4. Moyennage final 3 frames avec médiane anti-outlier (au lieu de moyenne sur 2).

        Paramètres :
          valid_ids  : liste d'IDs ArUco acceptés comme cibles (ex: [target_id_cible] ou ID_PIONS_ROBOT)
          z_target   : hauteur du plan cible en mm . Utilisé pour la déprojection pixel→robot
          h_scan     : hauteur de déplacement latéral pendant la correction (mm)
          is_stock   : True = cherche dans le stock (pion à saisir), False = cherche sur le plateau (case à déposer)
          ids_exclus : set d'IDs à ignorer (pions déjà posés sur le plateau)
          board_ia   : plateau logique . Nécessaire pour evaluer_rotation_pince_plateau()
          cible_idx  : index de la case cible . Nécessaire pour evaluer_rotation_pince_plateau()

        Retourne (pos_physique, eval_result) :
          - pos_physique : [X, Y] en mm dans le repère robot, calculé par pixel_vers_robot()
            NOTE : la correction mécanique est DEJA intégrée (ne pas la rajouter côté appelant).
          - eval_result  : yaw (int) si is_stock=True, (yaw, z_drop_offset) si is_stock=False

        DEBUG: si la caméra ne converge pas, vérifier :
          - offset_cam : décalage caméra/TCP bien configuré ?
          - dist_to_center : converge vers <SEUIL_PX ? Si non, le scale de pixel_vers_robot est mauvais.
        """
        if ids_exclus is None: ids_exclus = set()

        best_center = None   # (cx, cy) pixel de la meilleure cible trouvée dans la dernière itération
        best_corners = None  # Coins ArUco (4×2) de la meilleure cible . Utilisés pour calculer le yaw
        result_eval = None   # Résultat de evaluer_rotation_pince_* . Yaw ou (yaw, z_drop_offset)

        # [MODIF FIABILITE 2026-04-20] ID verrouillé : dès qu'on sélectionne une cible, on la garde
        # pour toutes les itérations suivantes (évite de changer de cible si plusieurs pions visibles).
        locked_id = None

        MAX_ITER = 5  # [MODIF VITESSE 2026-04-20] 6 -> 5 (correction mecanique + verrouillage ID = convergence plus rapide)
        SEUIL_PX = 8  # Seuil convergence pixel (~1-2 mm à h=300 mm)
        # [MODIF FIABILITE 2026-04-22] Flag de convergence : il faut avoir vu l'ArUco
        # centré à SEUIL_PX au moins une fois pour considerer la visee valide. Sinon on
        # rend None et le caller retombe sur son fallback (cases_memory ou pos_stock).
        # Sans ce garde-fou, une perte d'ArUco mid-iteration renvoyait une position
        # stale et le robot partait poser/saisir dans le vide.
        converged = False
        for step in range(MAX_ITER):
            vision.purger_buffer(2)  # [MODIF VITESSE 2026-04-20] 3 -> 2 frames
            img, corners, ids = vision.capturer()

            if img is not None:
                cv2.putText(img, _cv2txt(f"AUTOCORRECTION VISEE ({step+1}/{MAX_ITER})..."), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            if ids is None:
                # Aucun marqueur visible → abandon de la correction (position approximative)
                if img is not None:
                    cv2.imshow(win_name, img)
                    cv2.waitKey(1)
                break

            ids_flat = ids.flatten()  # IDs détectés dans ce frame (array 1D)
            cibles_infos = []  # Liste de tuples (mid, cx, cy, corners) pour les marqueurs dans valid_ids et non exclus
            all_centers = []   # Liste de (cx, cy) de tous les pions robot visibles (utilisé uniquement si is_stock=True)

            for i, mid in enumerate(ids_flat):
                if mid in ids_exclus: continue

                c = corners[i][0]
                cx, cy = np.mean(c[:, 0]), np.mean(c[:, 1])

                if is_stock and mid in valid_ids:
                    all_centers.append((cx, cy))
                    cibles_infos.append((int(mid), cx, cy, c))
                elif not is_stock and mid in valid_ids:
                    cibles_infos.append((int(mid), cx, cy, c))

            if not cibles_infos:
                # La cible a disparu du champ de vision → on garde la dernière bonne position
                if img is not None:
                    cv2.imshow(win_name, img)
                    cv2.waitKey(1)
                break

            # [MODIF FIABILITE 2026-04-20] Sélection cible avec verrouillage ID :
            # - si un ID a déjà été verrouillé et est toujours visible, on le garde
            # - sinon, on prend la cible la plus proche du centre image (choix initial)
            best_info = None
            if locked_id is not None:
                for info in cibles_infos:
                    if info[0] == locked_id:
                        best_info = info
                        break
            if best_info is None:
                best_info = min(cibles_infos, key=lambda p: math.hypot(p[1]-320, p[2]-240))
                locked_id = best_info[0]  # Verrouille l'ID pour les itérations suivantes

            best_center = (best_info[1], best_info[2])  # Centre pixel de la cible sélectionnée
            best_corners = best_info[3]                 # Coins ArUco (4×2) de la cible sélectionnée

            # Distance résiduelle entre la cible et le centre image (320, 240) en pixels
            dist_to_center = math.hypot(best_center[0]-320, best_center[1]-240)

            if is_stock:
                result_eval = self.evaluer_rotation_pince_stock(best_center, best_corners, all_centers)
                yaw_aff = result_eval
            else:
                result_eval = self.evaluer_rotation_pince_plateau(cible_idx, board_ia, best_corners)
                yaw_aff = result_eval[0]

            if img is not None:
                c0, c1, c2, c3 = best_corners
                pts = np.array([c0, c1, c2, c3], np.int32).reshape((-1, 1, 2))
                cv2.polylines(img, [pts], True, C_VERT, 2, cv2.LINE_AA)
                cv2.putText(img, _cv2txt(f"YAW: {yaw_aff} deg"), (int(best_center[0])-40, int(best_center[1])-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_VERT, 2, cv2.LINE_AA)
                cv2.imshow(win_name, img)
                cv2.waitKey(1)

            # Seuil de convergence : si la cible est déjà quasi-centrée, pas besoin de corriger
            if dist_to_center < SEUIL_PX:
                converged = True
                break

            # Calcule la position physique de la cible et déplace la caméra dessus
            pos_physique = vision.pixel_vers_robot(best_center[0], best_center[1], self.get_pose(), z_target=z_target)
            # La caméra est décalée par rapport au TCP → on soustrait l'offset pour déplacer LE TCP
            cam_target_x = pos_physique[0] - self.cfg["offset_cam"][0]
            cam_target_y = pos_physique[1] - self.cfg["offset_cam"][1]

            # [MODIF FIABILITE 2026-04-20] Hystérésis : si le déplacement demandé est infinitésimal, on sort.
            pose_avant = self.get_pose()
            delta_mvt = math.hypot(cam_target_x - pose_avant[0], cam_target_y - pose_avant[1])
            if delta_mvt < 1.5:  # < 1.5 mm = correction sub-résolution, inutile de bouger
                converged = True
                break

            self.deplacer(cam_target_x, cam_target_y, h_scan, speed=100, vision=vision, win_name=win_name, message="AJUSTEMENT VISEE...")
            self.sleep_visuel(WAIT_AJUSTEMENT_VISEE_S, vision, win_name)

        # [MODIF FIABILITE 2026-04-22] Si la convergence n'a jamais ete atteinte
        # (ArUco perdu avant le seuil de centrage), on rend None pour que le caller
        # bascule sur son fallback (cases_memory / pos_stock) au lieu de partir
        # viser une position stale.
        if not converged:
            print(f"[ROBOT] /!\\ Visee non convergee apres {MAX_ITER} iterations. Abandon.")
            return None, None

        if best_center:
            # [MODIF FIABILITE 2026-04-20] Calcul final : 3 frames + médiane anti-outlier.
            # La médiane rejette automatiquement une détection aberrante (1 frame sur 3 mauvais = OK).
            # Utilise locked_id si disponible pour rester sur la même cible pendant la moyenne.
            pose_finale = self.get_pose()
            positions_x = []
            positions_y = []
            target_id_avg = locked_id if locked_id is not None else None
            for _ in range(3):
                vision.purger_buffer(1)
                _, c2, i2 = vision.capturer()
                if i2 is not None:
                    ids2 = i2.flatten()
                    # Priorité : le locked_id exact, sinon 1er valide non-exclu
                    if target_id_avg is not None and target_id_avg in ids2:
                        idx2 = int(np.where(ids2 == target_id_avg)[0][0])
                    else:
                        valid = [mid for mid in ids2 if mid not in ids_exclus and mid in valid_ids] if valid_ids else list(ids2)
                        if not valid:
                            continue
                        idx2 = int(np.where(ids2 == valid[0])[0][0])
                    cx2, cy2 = np.mean(c2[idx2][0][:, 0]), np.mean(c2[idx2][0][:, 1])
                    p = vision.pixel_vers_robot(cx2, cy2, pose_finale, z_target=z_target)
                    if p:
                        positions_x.append(p[0])
                        positions_y.append(p[1])
            if positions_x:
                # Médiane au lieu de moyenne : rejette les outliers de détection
                pos = [float(np.median(positions_x)), float(np.median(positions_y))]
            else:
                pos = vision.pixel_vers_robot(best_center[0], best_center[1], pose_finale, z_target=z_target)
            return pos, result_eval

        return None, None

    def _verifier_correspondance_depot(self, vision, ids, corners, cases_memory):
        """Verifie que le pion robot detecte correspond bien a la case ArUco cible memorisee.

        Conditions de validation :
          1) coherence mapping ArUco case -> index de case,
          2) le marqueur ArUco de la case cible est masque (normal apres depot),
          3) un marqueur pion robot est au voisinage de cette case.

        Fallback : si le centre pixel de la case est indisponible, on compare en XY robot
        avec la position memoiree du lacher.
        """
        info = self.last_drop_info
        if not info or ids is None or corners is None:
            return False

        case_idx = int(info.get("case_idx", -1))
        aruco_case_id = int(info.get("aruco_case_id", -1))
        if case_idx < 0 or aruco_case_id < 0:
            return False

        if MAPPING_CASES.get(aruco_case_id, -999) != case_idx:
            return False

        ids_flat = ids.flatten()

        # Si l'ArUco de la case est encore visible, le pion ne couvre pas correctement la destination.
        if aruco_case_id in ids_flat:
            return False

        case_pos = cases_memory.get(case_idx)
        drop_xy = info.get("drop_xy")
        z_ref = float(info.get("z_ref", self.cfg["hauteurs"]["jeu"]))

        seuil_case_px = 70.0
        seuil_drop_mm = 45.0

        for i, mid in enumerate(ids_flat):
            if mid not in ID_PIONS_ROBOT:
                continue

            c = corners[i][0]
            pcx, pcy = int(np.mean(c[:, 0])), int(np.mean(c[:, 1]))

            # Verification principale: proximity pixel avec la case cible memorisee.
            if case_pos is not None:
                dist_case = math.hypot(pcx - case_pos[0], pcy - case_pos[1])
                if dist_case <= seuil_case_px:
                    info["detected_robot_aruco"] = int(mid)
                    info["detected_robot_px"] = (pcx, pcy)
                    return True
                continue

            # Fallback si on n'a plus le centre pixel de case: comparaison en repere robot.
            if drop_xy is not None:
                pos_robot = vision.pixel_vers_robot(pcx, pcy, self.get_pose(), z_target=z_ref)
                if pos_robot is not None:
                    dist_drop = math.hypot(pos_robot[0] - drop_xy[0], pos_robot[1] - drop_xy[1])
                    if dist_drop <= seuil_drop_mm:
                        info["detected_robot_aruco"] = int(mid)
                        info["detected_robot_xy"] = (float(pos_robot[0]), float(pos_robot[1]))
                        return True

        return False

    def sequence_poser_pion(self, vision, board_ia, coup_cible_idx, cases_memory, win_name, ids_exclus):
        """Séquence complète de prise + dépôt d'un pion du stock vers une case du plateau.

        Étapes :
          1. Recherche + saisie du pion dans le stock (3 essais max)
          2. Localisation précise de la case cible sur le plateau
          3. Dépôt du pion sur la case cible

        Retourne True si le pion a été posé, False si échec (3 saisies ratées).

        DEBUG: si la séquence échoue souvent en étape 1, vérifier :
          - pos_stock dans la config (robot ne voit pas le stock ?)
          - ID_PIONS_ROBOT non détectés → problème d'éclairage ou de config caméra
        """
        h_scan = self.cfg["pos_scan"][2]
        h_vol = self.cfg["hauteurs"]["vol"]
        h_jeu = self.cfg["hauteurs"]["jeu"]
        v_fast = self.cfg["params_mvt"]["speed_fast"]
        v_slow = self.cfg["params_mvt"]["speed_slow"]

        # --- ETAPE 1: RECHERCHE DANS LE STOCK (Portée réduite) ---
        p_stock = self.cfg["pos_stock"]
        pion_attrape = False
        pion_vu_dans_stock = False
        self.last_sequence_alert = ""
        range_recherche = 20

        for tentative in range(1, 4):  # 3 tentatives de saisie max
            target_pion_pos = None
            yaw_prise = 0

            # Point central de la caméra au-dessus du stock (offset caméra/TCP soustrait)
            cam_x_base = p_stock[0] - self.cfg["offset_cam"][0]
            cam_y_base = p_stock[1] - self.cfg["offset_cam"][1]

            range_recherche += 40*tentative  # Augmente la portée de recherche à chaque tentative (20mm, 70mm, 12q0mm)

            # PATROUILLE RESTREINTE: Faible portée (10mm) pour ratisser autour du stock
            ronde_offsets = [(0, 0), (range_recherche, 0), (0, range_recherche), (-range_recherche, 0), (0, -range_recherche)]

            for dx, dy in ronde_offsets:
                cam_x = cam_x_base + dx
                cam_y = cam_y_base + dy

                self.deplacer(cam_x, cam_y, h_scan, speed=v_fast, vision=vision, win_name=win_name, message="RECHERCHE STOCK")
                # [MODIF VITESSE 2026-04-20] 0.2 -> 0.12s (caméra stabilise vite à haute acc)
                self.sleep_visuel(WAIT_STOCK_ANALYSE_S, vision, win_name, "ANALYSE STOCK...")

                vision.purger_buffer(2)  # [MODIF VITESSE 2026-04-20] 3 -> 2 frames
                img, corners, ids = vision.capturer()

                if img is not None:
                    cv2.imshow(win_name, img)
                    cv2.waitKey(1)

                if ids is not None:
                    ids_flat = ids.flatten()
                    # Filtre : pions robot non déjà utilisés (ids_exclus = pions déjà sur le plateau)
                    pions_robot_valides = [m for m in ids_flat if m in ID_PIONS_ROBOT and m not in ids_exclus]

                    if pions_robot_valides:
                        # z_sol = hauteur réelle du sol du stock (plateau - épaisseur plateau)
                        z_sol = h_jeu - EPAISSEUR_PLATEAU
                        pos, yaw = self._centrer_camera_sur_cible(vision, ID_PIONS_ROBOT, z_sol, h_scan, win_name, is_stock=True, ids_exclus=ids_exclus)

                        if pos is not None:
                            target_pion_pos = pos
                            yaw_prise = yaw
                            pion_vu_dans_stock = True
                            break  # Pion trouvé et centré → sortir de la patrouille

            if target_pion_pos is None:
                print(f"[ROBOT] /!\\ Tentative {tentative}/3: aucun pion detecte dans le stock. Portée de recherche : {range_recherche} mm.")
                self.retour_scan_securise(vision, win_name)
                continue

            # Applique la correction mécanique (biais résiduel de calibration)
            target_pion_pos[0] += self.cfg["correction_mecanique_xy"][0]
            target_pion_pos[1] += self.cfg["correction_mecanique_xy"][1]

            # --- ETAPE 2: PRISE DU PION ---
            self.gripper(ouvrir=True, vision=vision, win_name=win_name)
            # Alignement horizontal d'abord (à hauteur scan = sûr)
            self.deplacer(target_pion_pos[0], target_pion_pos[1], h_scan, yaw=yaw_prise, speed=v_fast, vision=vision, win_name=win_name, message="ALIGNEMENT PINCE...")
            # Descente rapide jusqu'à la hauteur de vol
            self.deplacer(target_pion_pos[0], target_pion_pos[1], h_vol, yaw=yaw_prise, speed=v_fast, vision=vision, win_name=win_name, message="DESCENTE VOL...")

            # z_prise = hauteur à laquelle la pince doit se fermer pour saisir le pion
            # = sol du stock + moitié de la hauteur du pion (Z_PRISE_OFFSET = 25mm)
            z_sol = h_jeu - EPAISSEUR_PLATEAU
            z_prise = z_sol + Z_PRISE_OFFSET

            # Descente lente finale (vitesse slow pour plus de précision)
            self.deplacer(target_pion_pos[0], target_pion_pos[1], z_prise, yaw=yaw_prise, speed=v_slow, vision=vision, win_name=win_name, message="SAISIE...")

            # 1ère vérification : la pince ne s'est pas fermée à vide
            prise_ok = self.gripper(ouvrir=False, check_grab=True, vision=vision, win_name=win_name)

            # Remontée avec le pion (ou sans si la saisie a échoué)
            self.deplacer(target_pion_pos[0], target_pion_pos[1], h_scan, yaw=yaw_prise, speed=v_fast, vision=vision, win_name=win_name, message="REMONTEE...")

            # 2ème vérification : on n'a pas lâché le pion pendant la remontée
            if prise_ok and not self.simule and self.arm:
                code_pos, current_pos = self.arm.get_gripper_position()
                # Si < 50 = pince quasi-fermée = le pion est tombé pendant la remontée
                if code_pos == 0 and current_pos < 50:
                    prise_ok = False

            if not prise_ok:
                print("[ROBOT] /!\\ Echec de la prise ou pion perdu. Nouvel essai...")
                self.gripper(ouvrir=True, vision=vision, win_name=win_name)
                continue  # Tente la saisie à nouveau (tentative suivante)

            pion_attrape = True
            print("[ROBOT] Pion récupéré et sécurisé.")
            break

        if not pion_attrape:
            # 3 échecs consécutifs → abandon de la séquence
            if not pion_vu_dans_stock:
                self.last_sequence_alert = "ALERTE: aucun pion detecte dans le stock"
                print("[ROBOT] /!\\ ALERTE: stock vide ou hors champ apres plusieurs rondes.")
                self.retour_scan_securise(vision, win_name)
                self.sleep_visuel(1.2, vision, win_name, "ALERTE: PLUS DE PION DANS LE STOCK")
            else:
                self.last_sequence_alert = "ALERTE: echec de prise (3 essais)"
            return False

        # --- ETAPE 3: DÉTERMINATION DE LA CIBLE (PLATEAU) ---
        target_case_pos = None
        # Retrouve l'ID ArUco correspondant à la case cible (inverse de MAPPING_CASES)
        target_id_cible = CASE_TO_ARUCO.get(coup_cible_idx)
        if target_id_cible is None:
            self.last_sequence_alert = f"ALERTE: case cible invalide ({coup_cible_idx + 1})"
            print(f"[ROBOT] /!\\ Case cible invalide: {coup_cible_idx}")
            return False

        self.retour_scan_securise(vision, win_name)
        # [MODIF VITESSE 2026-04-20] 0.5 -> 0.25s (caméra stabilise vite après retour_scan)
        self.sleep_visuel(WAIT_PLATEAU_ANALYSE_S, vision, win_name, "ANALYSE DU PLATEAU...")

        # Tentative de localisation précise de la case par vision
        pos_cible, eval_plateau = self._centrer_camera_sur_cible(vision, [target_id_cible], h_jeu, h_scan, win_name, is_stock=False, board_ia=board_ia, cible_idx=coup_cible_idx)

        if pos_cible and eval_plateau is not None:
            target_case_pos = pos_cible
            yaw_depot, z_drop_offset = eval_plateau
        elif cases_memory[coup_cible_idx] is not None:
            # Fallback 1 : utilise la dernière position mémorisée de la case (moins précis)
            cx, cy = cases_memory[coup_cible_idx]
            target_case_pos = vision.pixel_vers_robot(cx, cy, self.get_pose(), z_target=h_jeu)
            yaw_depot, z_drop_offset = self.evaluer_rotation_pince_plateau(coup_cible_idx, board_ia)
        else:
            # Fallback 2 : estimation géométrique brute (très approximatif, dernier recours)
            target_case_pos = [300 + (coup_cible_idx%3 - 1)*60, (coup_cible_idx//3 - 1)*60, h_jeu]
            yaw_depot, z_drop_offset = self.evaluer_rotation_pince_plateau(coup_cible_idx, board_ia)

        # Applique la correction mécanique sur la cible plateau aussi
        target_case_pos[0] += self.cfg["correction_mecanique_xy"][0]
        target_case_pos[1] += self.cfg["correction_mecanique_xy"][1]

        # --- ETAPE 4: DÉPÔT DU PION ---
        # Alignement horizontal au-dessus de la case cible
        self.deplacer(target_case_pos[0], target_case_pos[1], h_scan, yaw=yaw_depot, speed=v_fast, vision=vision, win_name=win_name, message="ALIGNEMENT SUR CASE...")
        self.deplacer(target_case_pos[0], target_case_pos[1], h_vol, yaw=yaw_depot, speed=v_fast, vision=vision, win_name=win_name, message="DESCENTE VOL...")

        # Hauteur de lâcher = surface + offset de saisie + marge d'évitement des voisins
        # z_drop_offset est calculé par evaluer_rotation_pince_plateau selon la densité de voisins
        z_depot = h_jeu + Z_PRISE_OFFSET + z_drop_offset

        # Memoire explicite du dernier depot pour verifier la correspondance avec l'ArUco cible.
        self.last_drop_info = {
            "case_idx": int(coup_cible_idx),
            "aruco_case_id": int(target_id_cible),
            "drop_xy": (float(target_case_pos[0]), float(target_case_pos[1])),
            "z_ref": float(h_jeu),
            "timestamp": float(time.time()),
        }

        msg_drop = "DEPOT DU PION..."
        if z_drop_offset > 0:
            msg_drop = "LACHER EN HAUTEUR (SECURITE)..."

        self.deplacer(target_case_pos[0], target_case_pos[1], z_depot, yaw=yaw_depot, speed=v_slow, vision=vision, win_name=win_name, message=msg_drop)

        # Ouverture forcée (avec secousse si le pion reste collé)
        self.gripper(ouvrir=True, force_drop=True, vision=vision, win_name=win_name)
        # [MODIF VITESSE 2026-04-20] 0.3 -> 0.15s (pion tombe vite, pas besoin de longue pause)
        self.sleep_visuel(WAIT_POST_DROP_S, vision, win_name)

        self.deplacer(target_case_pos[0], target_case_pos[1], h_scan, yaw=yaw_depot, speed=v_fast, vision=vision, win_name=win_name, message="REMONTEE...")
        self.retour_scan_securise(vision, win_name)

        # Verification post-depot : ne valide le coup que si la case cible est bien occupee par O.
        # Sinon, on force un re-essai du cycle (retour stock -> nouvelle prise -> nouveau depot).
        confirmations = 0
        NB_CHECKS = 10
        for _ in range(NB_CHECKS):
            vision.purger_buffer(1)
            _, corners_chk, ids_chk = vision.capturer()
            phys_board_chk = evaluer_etat_physique(ids_chk, corners_chk, cases_memory)
            depot_sur_case = phys_board_chk[coup_cible_idx] == "O"
            correspondance_aruco = self._verifier_correspondance_depot(vision, ids_chk, corners_chk, cases_memory)

            if depot_sur_case and correspondance_aruco:
                confirmations += 1
                if confirmations >= 2:
                    break
            else:
                confirmations = 0

        if confirmations < 2:
            if self.last_drop_info:
                aruco_case = self.last_drop_info.get("aruco_case_id", target_id_cible)
                print(f"[ROBOT] /!\\ Depot non confirme sur case {coup_cible_idx + 1} (Aruco cible {aruco_case}). Reprise du cycle.")
            else:
                print(f"[ROBOT] /!\\ Depot non confirme sur case {coup_cible_idx + 1}. Reprise du cycle.")
            self.last_sequence_alert = f"ALERTE: depot non confirme case {coup_cible_idx + 1}"
            self.retour_scan_securise(vision, win_name)
            return False

        self.last_sequence_alert = ""
        return True


# ============================================================================
# 3. VISION OPTIMISEE & AFFICHAGE HUD 
# ============================================================================

def dessiner_hud( #OK
    img: np.ndarray,          # Frame couleur BGR (640×480) . Modifié en place (pas de copie)
    plateau: list[str],       # current_display_board : état logique à afficher (X/O/ )
    memoire_cases: dict,      # cases_memory : {index → (cx,cy) pixel} pour positionner les symboles
    tour_robot: bool,         # Détermine la couleur et le texte de la barre de statut
    presence_humain: dict,    # {index → compteur} pour la barre de progression du verrouillage
    simule: bool,             # True = fond noir + zones cliquables, False = flux caméra réel
    message_robot: str = "",  # Texte à afficher dans la barre de statut pendant le tour du robot
    seuil_validation: int = 30,    # Valeur max du compteur (100% de la barre de verrouillage)
    is_triche_mode: bool = False,  # True = erreur de cohérence plateau en cours de correction
    msg_triche: str = "",          # Description de l'erreur + consigne de réparation
    anomalies_triche=None,  # Liste de tuples (case_idx, attendu, vu)
    difficulte: str = "DIFFICILE"  # Niveau de l'IA à afficher dans le panneau latéral
) -> None:
    h, l = img.shape[:2]  # h = hauteur, l = largeur de l'image (attention: shape renvoie (hauteur, largeur))

    if simule:
        # En simulation : fond noir + zones de clic dessinées (pas de flux caméra réel)
        img[:] = (20, 20, 20)
        _dessiner_zones_cliquables(img, memoire_cases, plateau)

    # Limite horizontale de la zone de jeu (panneau latéral droit réservé)
    grid_right = l - 170

    # Marquage des pions sur la grille (gros + vifs pour visibilité caméra)
    for cidx in range(9):
        pos = memoire_cases.get(cidx)
        if pos is None:
            continue
        cx, cy = int(pos[0]), int(pos[1])
        val = plateau[cidx]

        # Ne dessine pas le contour des cases dans la zone du panneau latéral
        if cx > grid_right - 30:
            continue

        # Contour de case mémorisée un peu plus visible qu'avant
        cv2.rectangle(img, (cx - 34, cy - 34), (cx + 34, cy + 34), (70, 70, 70), 1, cv2.LINE_AA)

        if val == "X":
            _dessiner_croix(img, cx, cy, C_ORANGE, d=28, thick=5)
        elif val == "O":
            _dessiner_cercle(img, cx, cy, C_CYAN, r=30, thick=5)
        else:
            _dessiner_coins_vides(img, cx, cy, C_GRIS)
            _dessiner_verrouillage(img, cx, cy, presence_humain.get(cidx, 0), seuil_validation)

    # En cas d'erreur plateau, surligne visuellement les cases fautives.
    if is_triche_mode and anomalies_triche:
        pulse = int((math.sin(time.time() * 10.0) + 1.0) * 0.5 * 80)
        col = (0, 60 + pulse, 255)
        for cidx, attendu, vu in anomalies_triche:
            pos = memoire_cases.get(cidx)
            if pos is None:
                continue
            cx, cy = int(pos[0]), int(pos[1])
            cv2.circle(img, (cx, cy), 40, col, 3, cv2.LINE_AA)
            cv2.rectangle(img, (cx - 44, cy - 44), (cx + 44, cy + 44), col, 2, cv2.LINE_AA)
            attendu_txt = attendu if attendu != " " else "rien"
            vu_txt = vu if vu != " " else "rien"
            txt = f"att:{attendu_txt} vu:{vu_txt}"
            cv2.putText(img, _cv2txt(txt), (cx - 40, cy - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 2, cv2.LINE_AA)

    # --- Chrome UI ---
    _dessiner_barres_statut(img, l, h, tour_robot, simule, message_robot,
                            is_triche_mode=is_triche_mode, msg_triche=msg_triche,
                            difficulte=difficulte)
    _dessiner_panneau_score(img, l, h, plateau, tour_robot, message_robot,
                            presence_humain, seuil_validation, difficulte)
    _dessiner_bandeau_action(img, l, h, tour_robot, message_robot)


def dessiner_fin_de_partie(img: np.ndarray, gagnant: str) -> None: #OK
    h, w = img.shape[:2]
    overlay = img.copy()
    # Assombrit fortement l'image pour faire ressortir le panneau
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.72, img, 0.28, 0, img)

    if gagnant == "O":
        titre, sous, couleur, glyph = "VICTOIRE DU ROBOT", "BIP BOUP // SYSTEME IMBATTABLE", C_CYAN, "O"
    elif gagnant == "X":
        titre, sous, couleur, glyph = "VICTOIRE HUMAINE", "BIEN JOUE !", C_ORANGE, "X"
    else:
        titre, sous, couleur, glyph = "MATCH NUL", "AUCUN VAINQUEUR.", C_TENUE, "="

    # Panneau central HUD
    box_w, box_h = 460, 180
    bx = (w - box_w) // 2
    by = (h - box_h) // 2
    cv2.rectangle(img, (bx, by), (bx + box_w, by + box_h), (14, 14, 18), -1, cv2.LINE_AA)
    cv2.rectangle(img, (bx, by), (bx + box_w, by + box_h), couleur, 2, cv2.LINE_AA)

    # Coins type brackets
    bl = 22
    for (cx_, cy_, dx1, dy1, dx2, dy2) in [
        (bx, by, bl, 0, 0, bl),
        (bx + box_w, by, -bl, 0, 0, bl),
        (bx, by + box_h, bl, 0, 0, -bl),
        (bx + box_w, by + box_h, -bl, 0, 0, -bl),
    ]:
        cv2.line(img, (cx_, cy_), (cx_ + dx1, cy_ + dy1), couleur, 3, cv2.LINE_AA)
        cv2.line(img, (cx_, cy_), (cx_ + dx2, cy_ + dy2), couleur, 3, cv2.LINE_AA)

    # Bande d'entête
    cv2.rectangle(img, (bx, by), (bx + box_w, by + 28), couleur, -1)
    cv2.putText(img, _cv2txt("MORPION BIP BOUP  //  FIN DE PARTIE"),
                (bx + 14, by + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (12, 12, 12), 2, cv2.LINE_AA)

    # Glyph gagnant à gauche
    gcx, gcy = bx + 60, by + 110
    if glyph == "O":
        cv2.circle(img, (gcx, gcy), 34, couleur, 3, cv2.LINE_AA)
    elif glyph == "X":
        cv2.line(img, (gcx - 28, gcy - 28), (gcx + 28, gcy + 28), couleur, 3, cv2.LINE_AA)
        cv2.line(img, (gcx - 28, gcy + 28), (gcx + 28, gcy - 28), couleur, 3, cv2.LINE_AA)
    else:
        cv2.line(img, (gcx - 28, gcy - 8), (gcx + 28, gcy - 8), couleur, 3, cv2.LINE_AA)
        cv2.line(img, (gcx - 28, gcy + 8), (gcx + 28, gcy + 8), couleur, 3, cv2.LINE_AA)

    # Titre + sous-titre
    text_x = bx + 120
    (tw, _), _ = cv2.getTextSize(titre, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
    cv2.putText(img, _cv2txt(titre), (text_x, by + 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, couleur, 3, cv2.LINE_AA)
    cv2.line(img, (text_x, by + 112), (text_x + tw, by + 112), couleur, 1, cv2.LINE_AA)
    cv2.putText(img, _cv2txt(sous), (text_x, by + 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_TENUE, 1, cv2.LINE_AA)

    # Pied : horodatage
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(img, _cv2txt(f">> {stamp}"), (bx + 14, by + box_h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_TENUE, 1, cv2.LINE_AA)


def _dessiner_barres_statut(img: np.ndarray, l: int, h: int, tour_robot: bool, simule: bool,
                            message_robot: str, is_triche_mode: bool = False,
                            msg_triche: str = "", difficulte: str = "DIFFICILE") -> None: #OK
    """Bandeau supérieur pleine largeur : indicateur de tour géant, minimaliste."""
    TOP_H = 72

    # Fond opaque
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (l, TOP_H), (14, 14, 16), -1)
    cv2.addWeighted(overlay, 0.92, img, 0.08, 0, img)

    if is_triche_mode:
        couleur = C_ROUGE
        glyph = "!"
        label = "CORRIGEZ LE PLATEAU"
    elif tour_robot:
        couleur = C_CYAN
        glyph = "O"
        label = "ROBOT JOUE"
    else:
        couleur = C_ORANGE
        glyph = "X"
        label = "A VOUS DE JOUER"

    # Liseré bas unique (sobriété)
    cv2.rectangle(img, (0, TOP_H - 3), (l, TOP_H), couleur, -1)

    # Glyph centré vertical
    gcx, gcy = 44, TOP_H // 2
    if glyph == "O":
        cv2.circle(img, (gcx, gcy), 22, couleur, 4, cv2.LINE_AA)
    elif glyph == "!":
        cv2.rectangle(img, (gcx - 16, gcy - 22), (gcx + 16, gcy + 22), couleur, 3, cv2.LINE_AA)
        cv2.line(img, (gcx, gcy - 12), (gcx, gcy + 6), couleur, 4, cv2.LINE_AA)
        cv2.circle(img, (gcx, gcy + 14), 2, couleur, -1, cv2.LINE_AA)
    else:
        cv2.line(img, (gcx - 20, gcy - 20), (gcx + 20, gcy + 20), couleur, 5, cv2.LINE_AA)
        cv2.line(img, (gcx - 20, gcy + 20), (gcx + 20, gcy - 20), couleur, 5, cv2.LINE_AA)

    # Label clair, taille maîtrisée
    (_, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
    cv2.putText(img, _cv2txt(label), (84, gcy + th // 2 - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, couleur, 3, cv2.LINE_AA)

    if is_triche_mode and msg_triche:
        txt = msg_triche[:70]
        cv2.putText(img, _cv2txt(txt), (84, TOP_H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)

    # Badges en haut à droite : niveau a gauche de SIM en simulation,
    # sinon niveau a la place de SIM en mode reel.
    badge_x = l - 60
    badge_w = 48
    badge_h = 18
    badge_gap = 6

    niveau_label = {"FACILE": "N1", "MOYEN": "N2", "DIFFICILE": "N3"}.get(difficulte, "N?")
    niveau_col = {"FACILE": C_VERT, "MOYEN": C_ORANGE, "DIFFICILE": C_CYAN}.get(difficulte, C_TENUE)

    def _draw_badge(x: int, label: str, txt_col: tuple[int, int, int] = C_TENUE) -> None:
        cv2.rectangle(img, (x, 10), (x + badge_w, 10 + badge_h), (30, 30, 34), -1, cv2.LINE_AA)
        cv2.rectangle(img, (x, 10), (x + badge_w, 10 + badge_h), (80, 80, 88), 1, cv2.LINE_AA)
        cv2.putText(img, _cv2txt(label), (x + 10, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, txt_col, 1, cv2.LINE_AA)

    if simule:
        niveau_x = badge_x - (badge_w + badge_gap)
        _draw_badge(niveau_x, niveau_label, niveau_col)
        _draw_badge(badge_x, "SIM")
    else:
        _draw_badge(badge_x, niveau_label, niveau_col)


def _dessiner_panneau_score(img: np.ndarray, l: int, h: int, plateau: list[str],
                            tour_robot: bool, message_robot: str,
                            presence_humain: dict, seuil_validation: int,
                            difficulte: str = "DIFFICILE") -> None:
    """Panneau latéral droit : tour, score, pions restants, progression, dernier coup."""
    SIDE_W = 170
    TOP_H = 72
    BOT_H = 56
    x0 = l - SIDE_W
    y0 = TOP_H + 8
    y1 = h - BOT_H - 8

    # Fond panneau
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (l, y1), (14, 14, 16), -1)
    cv2.addWeighted(overlay, 0.90, img, 0.10, 0, img)
    cv2.rectangle(img, (x0, y0), (l, y1), (60, 60, 60), 1, cv2.LINE_AA)

    accent = C_ORANGE if not tour_robot else C_CYAN
    cv2.rectangle(img, (x0, y0), (x0 + 3, y1), accent, -1)

    nb_x = sum(1 for c in plateau if c == "X")
    nb_o = sum(1 for c in plateau if c == "O")
    tour_num = nb_x + nb_o + 1

    PAD_L = x0 + 12
    PAD_R = l - 12

    # =========================================================================
    # BLOC 1 : TOUR courant (gros chiffre + qui joue)
    # =========================================================================
    y = y0 + 18
    cv2.putText(img, _cv2txt("TOUR"), (PAD_L, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_TENUE, 1, cv2.LINE_AA)
    # Chiffre gros à gauche + fraction
    big = f"{tour_num}"
    cv2.putText(img, _cv2txt(big), (PAD_L, y + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (240, 240, 240), 2, cv2.LINE_AA)
    (bw, _), _ = cv2.getTextSize(big, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    cv2.putText(img, _cv2txt("/9"), (PAD_L + bw + 4, y + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_TENUE, 1, cv2.LINE_AA)

    # Pastille actif à droite (X ou O selon joueur en cours)
    tag_cx = PAD_R - 18
    tag_cy = y + 24
    if tour_robot:
        cv2.circle(img, (tag_cx, tag_cy), 13, C_CYAN, 3, cv2.LINE_AA)
    else:
        cv2.line(img, (tag_cx - 11, tag_cy - 11), (tag_cx + 11, tag_cy + 11), C_ORANGE, 3, cv2.LINE_AA)
        cv2.line(img, (tag_cx - 11, tag_cy + 11), (tag_cx + 11, tag_cy - 11), C_ORANGE, 3, cv2.LINE_AA)

    # Barre de progression de la partie
    bar_y = y + 44
    cv2.rectangle(img, (PAD_L, bar_y), (PAD_R, bar_y + 4), (40, 40, 44), -1)
    prog_w = int((min(tour_num - 1, 9) / 9.0) * (PAD_R - PAD_L))
    if prog_w > 0:
        cv2.rectangle(img, (PAD_L, bar_y), (PAD_L + prog_w, bar_y + 4), accent, -1)

    # =========================================================================
    # BLOC 2 : SCORE compact (2 lignes)
    # =========================================================================
    y = bar_y + 20
    cv2.line(img, (PAD_L, y), (PAD_R, y), (45, 45, 50), 1, cv2.LINE_AA)
    y += 16
    cv2.putText(img, _cv2txt("SCORE"), (PAD_L, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_TENUE, 1, cv2.LINE_AA)

    def _score_line(row_y: int, glyph: str, nom: str, val: int, col: tuple):
        gx = PAD_L + 8
        if glyph == "X":
            d = 7
            cv2.line(img, (gx - d, row_y - d), (gx + d, row_y + d), col, 2, cv2.LINE_AA)
            cv2.line(img, (gx - d, row_y + d), (gx + d, row_y - d), col, 2, cv2.LINE_AA)
        else:
            cv2.circle(img, (gx, row_y), 7, col, 2, cv2.LINE_AA)
        cv2.putText(img, _cv2txt(nom), (PAD_L + 24, row_y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 225), 1, cv2.LINE_AA)
        val_txt = str(val)
        (vw, _), _ = cv2.getTextSize(val_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(img, _cv2txt(val_txt), (PAD_R - vw, row_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

    _score_line(y + 20, "X", "HUMAIN", nb_x, C_ORANGE)
    _score_line(y + 42, "O", "ROBOT",  nb_o, C_CYAN)

    # =========================================================================
    # BLOC 3 : PIONS restants (visuellement clair : points colorés)
    # =========================================================================
    y = y + 62
    cv2.line(img, (PAD_L, y), (PAD_R, y), (45, 45, 50), 1, cv2.LINE_AA)
    y += 16
    cv2.putText(img, _cv2txt("PIONS RESTANTS"), (PAD_L, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_TENUE, 1, cv2.LINE_AA)

    # Humain : 5 pions max, O vide = joué
    total_x = 5
    restant_x = max(0, total_x - nb_x)
    for i in range(total_x):
        dot_x = PAD_L + 6 + i * 14
        dot_y = y + 18
        if i < restant_x:
            cv2.circle(img, (dot_x, dot_y), 4, C_ORANGE, -1, cv2.LINE_AA)
        else:
            cv2.circle(img, (dot_x, dot_y), 4, (50, 50, 55), 1, cv2.LINE_AA)

    # Robot : 4 pions max (il joue second)
    total_o = 4
    restant_o = max(0, total_o - nb_o)
    for i in range(total_o):
        dot_x = PAD_L + 6 + i * 14
        dot_y = y + 36
        if i < restant_o:
            cv2.circle(img, (dot_x, dot_y), 4, C_CYAN, -1, cv2.LINE_AA)
        else:
            cv2.circle(img, (dot_x, dot_y), 4, (50, 50, 55), 1, cv2.LINE_AA)

    # =========================================================================
    # BLOC 4 : STATUT coup en cours (pour humain : progression verrouillage
    #          max sur toutes les cases ; pour robot : spinner réflexion)
    # =========================================================================
    y = y + 54
    cv2.line(img, (PAD_L, y), (PAD_R, y), (45, 45, 50), 1, cv2.LINE_AA)
    y += 16
    cv2.putText(img, _cv2txt("STATUT"), (PAD_L, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_TENUE, 1, cv2.LINE_AA)

    if tour_robot:
        # Spinner 3 points animés
        t = time.time()
        msg = "Reflexion"
        cv2.putText(img, _cv2txt(msg), (PAD_L, y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_CYAN, 1, cv2.LINE_AA)
        (mw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        for i in range(3):
            phase = (t * 2.5 + i * 0.35) % 1.0
            alpha = 0.3 + (1 - abs(phase - 0.5) * 2) * 0.7
            col = tuple(int(c * alpha) for c in C_CYAN)
            cv2.circle(img, (PAD_L + mw + 8 + i * 8, y + 18), 2, col, -1, cv2.LINE_AA)
    else:
        # Verrouillage en cours : % max sur toutes les cases
        max_compteur = max(presence_humain.values()) if presence_humain else 0
        if max_compteur > 0:
            pct = min(100, int((max_compteur / float(seuil_validation)) * 100))
            cv2.putText(img, _cv2txt("Verrouillage"), (PAD_L, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 225), 1, cv2.LINE_AA)
            cv2.putText(img, _cv2txt(f"{pct}%"), (PAD_R - 36, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_ORANGE, 2, cv2.LINE_AA)
            # Mini-barre
            mb_y = y + 28
            cv2.rectangle(img, (PAD_L, mb_y), (PAD_R, mb_y + 4), (40, 40, 44), -1)
            mb_w = int((pct / 100.0) * (PAD_R - PAD_L))
            cv2.rectangle(img, (PAD_L, mb_y), (PAD_L + mb_w, mb_y + 4), C_ORANGE, -1)
        else:
            cv2.putText(img, _cv2txt("En attente"), (PAD_L, y + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_TENUE, 1, cv2.LINE_AA)

    # =========================================================================
    # BLOC 5 : DERNIER COUP (détecté via snapshot inter-frames)
    # =========================================================================
    # Snapshot du plateau précédent stocké sur la fonction elle-même
    prev = getattr(_dessiner_panneau_score, "_prev_plateau", [" "] * 9)
    dernier = getattr(_dessiner_panneau_score, "_dernier", None)  # (case, joueur)
    for i in range(9):
        if plateau[i] != " " and prev[i] == " ":
            dernier = (i + 1, plateau[i])
            _dessiner_panneau_score._dernier = dernier
            break
    _dessiner_panneau_score._prev_plateau = list(plateau)

    y = y + 44
    cv2.line(img, (PAD_L, y), (PAD_R, y), (45, 45, 50), 1, cv2.LINE_AA)
    y += 16
    cv2.putText(img, _cv2txt("DERNIER COUP"), (PAD_L, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_TENUE, 1, cv2.LINE_AA)

    if dernier is None:
        cv2.putText(img, _cv2txt("-"), (PAD_L, y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 95), 1, cv2.LINE_AA)
    else:
        case_num, joueur = dernier
        col_joueur = C_ORANGE if joueur == "X" else C_CYAN
        # Glyph + "Case N"
        gx, gy = PAD_L + 8, y + 20
        if joueur == "X":
            d = 7
            cv2.line(img, (gx - d, gy - d), (gx + d, gy + d), col_joueur, 2, cv2.LINE_AA)
            cv2.line(img, (gx - d, gy + d), (gx + d, gy - d), col_joueur, 2, cv2.LINE_AA)
        else:
            cv2.circle(img, (gx, gy), 7, col_joueur, 2, cv2.LINE_AA)
        cv2.putText(img, _cv2txt(f"Case {case_num}"), (PAD_L + 24, y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 225), 1, cv2.LINE_AA)

    # =========================================================================
    # BLOC 6 : NIVEAU IA (3 pastilles : FACILE / MOYEN / DIFFICILE)
    # =========================================================================
    y = y + 44
    cv2.line(img, (PAD_L, y), (PAD_R, y), (45, 45, 50), 1, cv2.LINE_AA)
    y += 16
    cv2.putText(img, _cv2txt("NIVEAU"), (PAD_L, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_TENUE, 1, cv2.LINE_AA)

    # 3 segments horizontaux, celui du niveau courant est surligné
    seg_y = y + 14
    seg_h = 14
    seg_w = (PAD_R - PAD_L - 4) // 3
    niveaux = (("FACILE", "1"), ("MOYEN", "2"), ("DIFFICILE", "3"))
    for i, (nom, touche) in enumerate(niveaux):
        sx = PAD_L + i * (seg_w + 2)
        actif = (nom == difficulte)
        col_bg = accent if actif else (38, 38, 42)
        col_txt = (14, 14, 16) if actif else (200, 200, 205)
        cv2.rectangle(img, (sx, seg_y), (sx + seg_w, seg_y + seg_h), col_bg, -1)
        label = touche  # chiffre seul : tient dans le segment
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.putText(img, _cv2txt(label),
                    (sx + (seg_w - tw) // 2, seg_y + (seg_h + th) // 2 - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col_txt, 1, cv2.LINE_AA)
    # Nom du niveau sous les segments
    name_y = seg_y + seg_h + 14
    cv2.putText(img, _cv2txt(difficulte), (PAD_L, name_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, accent, 1, cv2.LINE_AA)

    # =========================================================================
    # BLOC 7 : Raccourci clavier (zone dédiée en haut à droite)
    # Evite la superposition avec "DERNIER COUP" quand le panneau est dense.
    # =========================================================================
    key_label = "[Q] QUITTER"
    (kw, _), _ = cv2.getTextSize(key_label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    key_x = PAD_R - kw
    key_y = y0 + 14
    cv2.putText(img, _cv2txt(key_label), (key_x, key_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_TENUE, 1, cv2.LINE_AA)


def _dessiner_bandeau_action(img: np.ndarray, l: int, h: int, tour_robot: bool,
                             message_robot: str) -> None:
    """Bandeau d'action en bas : instruction courte, une seule ligne."""
    BOT_H = 56

    overlay = img.copy()
    cv2.rectangle(overlay, (0, h - BOT_H), (l, h), (10, 10, 12), -1)
    cv2.addWeighted(overlay, 0.92, img, 0.08, 0, img)

    if tour_robot:
        couleur = C_CYAN
        instruction = (message_robot or "LE ROBOT REFLECHIT").upper()
    else:
        couleur = C_ORANGE
        instruction = "POSEZ VOTRE PION SUR UNE CASE VIDE"

    cv2.rectangle(img, (0, h - BOT_H), (l, h - BOT_H + 3), couleur, -1)

    # Instruction, scale adaptatif, centrée verticalement
    max_w = l - 60
    scale = 0.9
    for s in (0.9, 0.8, 0.7, 0.6, 0.5):
        (w_txt, _), _ = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, s, 2)
        if w_txt <= max_w:
            scale = s
            break
    (_, th), _ = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    ty = h - BOT_H // 2 + th // 2 - 2
    cv2.putText(img, _cv2txt(instruction), (24, ty),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (240, 240, 240), 2, cv2.LINE_AA)


def _pixel_vers_xy_simu(cx: int, cy: int) -> tuple[int, int]: #OK
    """Calcule les coordonnées physiques XY (mm, repère robot) d'un pixel en mode simulation.
    Utilise la même formule que VisionSystem.pixel_vers_robot() en mode simulé,
    avec pos_scan comme pose de référence (caméra au-dessus du plateau)."""
    offsets   = cfg.data["offset_cam"]
    scan_pose = cfg.data["pos_scan"]
    scale     = 0.5  # mm par pixel (approximation simulation)
    dx = (cx - 320) * scale
    dy = (cy - 240) * scale
    x_mm = scan_pose[0] + offsets[0] + dy
    y_mm = scan_pose[1] + offsets[1] - dx
    return int(round(x_mm)), int(round(y_mm))


def _dessiner_zones_cliquables(img: np.ndarray, memoire_cases: dict, plateau: list[str]) -> None: #OK
    """En mode simulation : dessine la grille complète avec coordonnées XY robot (mm)."""
    demi = 46  # Demi-taille d'une case en pixels

    # Récupère les centres de colonnes (cases 0,1,2) et de lignes (cases 0,3,6)
    col_xs = [memoire_cases.get(j) for j in [0, 1, 2] if memoire_cases.get(j) is not None]
    row_ys = [memoire_cases.get(i * 3) for i in [0, 1, 2] if memoire_cases.get(i * 3) is not None]

    if col_xs and row_ys:
        x_left  = int(col_xs[0][0])  - demi
        x_right = int(col_xs[-1][0]) + demi
        y_top   = int(row_ys[0][1])  - demi
        y_bot   = int(row_ys[-1][1]) + demi

        # Cadre extérieur + lignes de séparation
        cv2.rectangle(img, (x_left, y_top), (x_right, y_bot), (100, 100, 100), 2, cv2.LINE_AA)
        for j in range(len(col_xs) - 1):
            sep_x = (int(col_xs[j][0]) + int(col_xs[j + 1][0])) // 2
            cv2.line(img, (sep_x, y_top), (sep_x, y_bot), (100, 100, 100), 2, cv2.LINE_AA)
        for i in range(len(row_ys) - 1):
            sep_y = (int(row_ys[i][1]) + int(row_ys[i + 1][1])) // 2
            cv2.line(img, (x_left, sep_y), (x_right, sep_y), (100, 100, 100), 2, cv2.LINE_AA)

        # Légende Z (commune à toutes les cases) sous la grille
        z_mm = int(round(cfg.data["hauteurs"]["jeu"]))
        cv2.putText(img, _cv2txt(f"Z = {z_mm} mm"),
                    (x_left, y_bot + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (70, 70, 70), 1, cv2.LINE_AA)

    # Cases individuelles : numéro de case au centre
    for cidx in range(9):
        pos = memoire_cases.get(cidx)
        if pos is None:
            continue
        cx, cy = int(pos[0]), int(pos[1])

        if plateau[cidx] == " ":
            # Fond légèrement éclairé . Case cliquable
            cv2.rectangle(img, (cx - demi + 3, cy - demi + 3), (cx + demi - 3, cy + demi - 3),
                          (35, 35, 35), -1, cv2.LINE_AA)
            # Numéro de case (1-9) centré
            num = str(cidx + 1)
            (tw, th), _ = cv2.getTextSize(num, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.putText(img, num, (cx - tw // 2, cy + th // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 80), 2, cv2.LINE_AA)

    # Panneau "Cibles robot" supprimé de l'UI de jeu . Réservé au debug calibration.
    # (Les coordonnées sont toujours calculables via _pixel_vers_xy_simu si besoin.)

def _dessiner_croix(img: np.ndarray, cx: int, cy: int, couleur: tuple, d: int = 22, thick: int = 3) -> None: #OK
    cv2.line(img, (cx - d, cy - d), (cx + d, cy + d), couleur, thick, cv2.LINE_AA)
    cv2.line(img, (cx - d, cy + d), (cx + d, cy - d), couleur, thick, cv2.LINE_AA)

def _dessiner_cercle(img: np.ndarray, cx: int, cy: int, couleur: tuple, r: int = 26, thick: int = 3) -> None: #OK
    cv2.circle(img, (cx, cy), r, couleur, thick, cv2.LINE_AA)

def _dessiner_coins_vides(img: np.ndarray, cx: int, cy: int, couleur: tuple) -> None: #OK
    """Dessine 4 petits coins pour indiquer une case vide (style UI moderne)."""
    r, lg = 30, 9
    for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        ox, oy = cx + sx * r, cy + sy * r
        cv2.line(img, (ox, oy), (ox - sx * lg, oy), couleur, 2, cv2.LINE_AA)
        cv2.line(img, (ox, oy), (ox, oy - sy * lg), couleur, 2, cv2.LINE_AA)

def _dessiner_verrouillage(img: np.ndarray, cx: int, cy: int, compteur: int, seuil: int) -> None: #OK
    """Anneau de confirmation clair : piste + arc + compte à rebours central.
    Couleur = C_ORANGE (couleur joueur humain, cohérent avec "à vous de jouer")."""
    if compteur <= 0: return

    ratio = min(1.0, compteur / float(seuil))
    couleur = C_ORANGE
    RAYON = 32

    # --- Piste arrière épaisse (très lisible) ---
    cv2.circle(img, (cx, cy), RAYON, (45, 45, 50), 4, cv2.LINE_AA)

    # --- Arc de progression : épais, tracé par petits segments pour lissage ---
    angle_fin = int(360 * ratio)
    if angle_fin > 0:
        cv2.ellipse(img, (cx, cy), (RAYON, RAYON), -90, 0, angle_fin,
                    couleur, 4, cv2.LINE_AA)

    # --- Pastille centrale avec compte à rebours ---
    # Temps restant en secondes (seuil ≈ 30 frames à 30fps → 1s)
    restant = max(0.0, (seuil - compteur) / 30.0)
    label = f"{restant:.1f}" if restant > 0 else "OK"

    # Fond pastille
    cv2.circle(img, (cx, cy), 18, (18, 18, 22), -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 18, couleur, 1, cv2.LINE_AA)

    # Texte centré
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.putText(img, _cv2txt(label), (cx - tw // 2, cy + th // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, couleur, 2, cv2.LINE_AA)

    # --- Flash de validation à 100% : anneau blanc pulsé ---
    if ratio >= 1.0:
        pulse = (math.sin(time.time() * 12) + 1) / 2
        flash_r = RAYON + 6 + int(pulse * 4)
        cv2.circle(img, (cx, cy), flash_r, (255, 255, 255), 2, cv2.LINE_AA)

class VisionSystem:
    def __init__(self): #OK
        self.simule = True          # True si RealSense indisponible. Capturer() renvoie un frame noir
        self.pipeline = None        # rs.pipeline . Objet de flux caméra RealSense, None si non initialisé
        self.intrinsics = None      # rs.intrinsics . Paramètres optiques (fx, fy, cx, cy, distortion)
                                    # Utilisé par rs2_deproject_pixel_to_point() dans pixel_vers_robot()
        self.detector = None        # cv2.aruco.ArucoDetector . Détecteur configuré une seule fois au démarrage
        self.click_coords = None    # (x, y) pixel du dernier clic gauche . Consommé dans la boucle principale
                                    # Mis à None après lecture pour éviter le double-déclenchement
        self.frame_count = 0        # Compteur de frames capturés depuis le démarrage (debug uniquement)
        self.last_valid_frame = None  # Dernier frame couleur reçu avec succès . Fallback en cas de timeout RealSense

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        detector_params = cv2.aruco.DetectorParameters()

        # Paramètres de détection ArUco . Optimisés pour marqueurs petits et éclairage variable
        # DEBUG: si des marqueurs ne sont pas détectés, essayer de réduire minMarkerPerimeterRate
        #        ou augmenter adaptiveThreshWinSizeMax
        detector_params.adaptiveThreshConstant = 7          # Constante du seuil adaptatif (C dans T = mean - C)
        detector_params.adaptiveThreshWinSizeMin = 3        # Taille min de la fenêtre de seuillage (px, impair)
        detector_params.adaptiveThreshWinSizeMax = 33       # Taille max . Plus grand = détecte les marqueurs plus larges
        detector_params.adaptiveThreshWinSizeStep = 10      # Pas entre tailles testées (3, 13, 23, 33)
        detector_params.polygonalApproxAccuracyRate = 0.05  # Tolérance de l'approximation polygonale (5% du périmètre)
        detector_params.minMarkerPerimeterRate = 0.03       # Périmètre min accepté = 3% de la diagonale image (~19px sur 640px)
        detector_params.maxMarkerPerimeterRate = 4.0        # Périmètre max = 400% de la diagonale (pas de limite pratique)
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

        if REALSENSE_AVAILABLE:
            print("[VISION] Tentative d'initialisation RealSense...")
            try:
                self.pipeline = rs.pipeline()
                config = rs.config()
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                profile = self.pipeline.start(config)
                # Chauffe de la caméra : 30 frames jetés pour stabiliser l'exposition
                for i in range(30):
                    try: self.pipeline.wait_for_frames(timeout_ms=1000)
                    except: pass

                # Récupère les intinsèques pour la déprojection pixel→point 3D
                self.intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
                self.simule = False
                print("[VISION] CAMERA REALSENSE DEMARREE AVEC SUCCES")
            except Exception as e:
                print("[VISION] ERREUR CRITIQUE REALSENSE : {}".format(e))
                self.simule = True
                self.pipeline = None
        else:
            print("[VISION] Module pyrealsense2 non installe.")

    def set_mouse_callback(self, win_name): #OK
        """Enregistre le callback clic pour la fenêtre OpenCV (utilisé en mode simulation)."""
        def mouse_handler(event, x, y, flags, param):
            if self.simule and event == cv2.EVENT_LBUTTONDOWN:
                self.click_coords = (x, y)
        cv2.setMouseCallback(win_name, mouse_handler)

    def appliquer_gamma(self, image, gamma=2.0): #OK
        """Correction gamma via lookup table (LUT) . Plus rapide qu'un calcul pixel par pixel.
        gamma > 1 : éclaircit les zones sombres (utile si les marqueurs sont sous-exposés)."""
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def get_processed_image(self, color_img): #OK
        """Pipeline de prétraitement pour améliorer la détection ArUco :
          1. Correction gamma (luminosité)
          2. Conversion en niveaux de gris
          3. CLAHE (égalisation adaptative du contraste local)
        Retourne une image en niveaux de gris rehaussée."""
        gamma = cfg.data["vision"]["gamma"]
        clip = cfg.data["vision"]["clahe_clip"]
        img_gamma = self.appliquer_gamma(color_img, gamma=gamma)
        gray = cv2.cvtColor(img_gamma, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def capturer(self): #OK
        """Capture un frame RealSense et détecte les marqueurs ArUco.

        Stratégie triple détection (pour maximiser la détection de l'ArUco 0 notamment) :
          1. Détection sur l'image prétraitée (gamma + CLAHE) . Meilleure dans la plupart des cas
          2. Fallback sur l'image brute couleur si la 1ère détection échoue
          3. Fallback sur l'image avec filtre de netteté + niveaux de gris purs (sans CLAHE)
             → utile pour ArUco 0 dont le pattern dense est dégradé par le CLAHE

        Retourne (color_img, corners, ids) . Corners et ids sont None si aucun marqueur détecté.
        DEBUG: si frame_count monte mais ids reste None, vérifier l'éclairage ou les paramètres ArUco.
        """
        color_img = None
        corners = None
        ids = None

        if not self.simule and self.pipeline:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=100)
                c = frames.get_color_frame()
                if c:
                    color_img = np.asanyarray(c.get_data())
                    self.last_valid_frame = color_img.copy()  # Sauvegarde pour le fallback timeout
            except Exception as e:
                # Timeout ou erreur caméra → utilise le dernier frame valide
                if self.last_valid_frame is not None: color_img = self.last_valid_frame
                else: color_img = None

        # En simulation (ou si caméra indisponible) : frame noir 640x480
        if color_img is None: color_img = np.zeros((480, 640, 3), dtype=np.uint8)

        if color_img is not None:
            try:
                gray_contrast = self.get_processed_image(color_img)
                # Tentative 1 : image prétraitée (meilleure détection en général)
                corners, ids, _ = self.detector.detectMarkers(gray_contrast)
                if ids is None:
                    # Tentative 2 : image couleur brute (fallback si prétraitement dégrade)
                    corners, ids, _ = self.detector.detectMarkers(color_img)
                if ids is None:
                    # Tentative 3 : sharpening + niveaux de gris purs
                    # Le filtre de netteté renforce les bords du marqueur sans modifier la luminosité globale.
                    # Particulièrement efficace pour l'ArUco 0 dont le contraste bord/fond peut être faible.
                    kernel_sharp = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
                    img_sharp = cv2.filter2D(color_img, -1, kernel_sharp)
                    img_sharp = np.clip(img_sharp, 0, 255).astype(np.uint8)
                    gray_sharp = cv2.cvtColor(img_sharp, cv2.COLOR_BGR2GRAY)
                    corners, ids, _ = self.detector.detectMarkers(gray_sharp)
            except Exception as e:
                corners = None
                ids = None

        self.frame_count += 1
        return color_img, corners, ids

    def purger_buffer(self, nb_frames=15): #OK
        """Vide nb_frames frames de la file RealSense pour obtenir une image fraîche.
        Utile avant une détection critique (évite d'analyser un frame vieux de 0.5s)."""
        if self.simule or not self.pipeline: return
        for _ in range(nb_frames):
            try: self.pipeline.wait_for_frames(timeout_ms=100)
            except: pass

    def pixel_vers_robot(self, u, v, robot_pose, z_target=None): #OK
        """Convertit des coordonnées pixel (u, v) en position physique (X, Y) dans le repère robot (mm).

        En mode réel :
          Utilise rs2_deproject_pixel_to_point avec la distance caméra→plan cible.
          dist_m = (Z_caméra - Z_cible) / 1000   [en mètres]

        En mode simulé :
          Approximation linéaire simple avec scale=0.5 px/mm.

        Convention axes (réel) :
          cam_x (latéral caméra) → vec_y_robot (Y robot)
          cam_y (profondeur caméra) → vec_x_robot (X robot)
          NOTE : les axes X/Y caméra sont permutés par rapport au repère robot.

        DEBUG: si les positions calculées sont décalées mais pas de biais constant :
          → vérifier z_target (hauteur du plan cible incorrecte)
          → vérifier offset_cam (décalage caméra/TCP)
        """
        offsets = cfg.data["offset_cam"]

        if self.simule or not self.intrinsics:
            # Simulation : relation linéaire simplifiée centrée sur (320, 240)
            scale = 0.5  # mm par pixel (approximation grossière)
            dx = (u - 320) * scale
            dy = (v - 240) * scale
            pos_x = robot_pose[0] + offsets[0] + dy  # dy pixel → direction X robot
            pos_y = robot_pose[1] + offsets[1] - dx  # dx pixel → direction Y robot (inversé)
            return [pos_x, pos_y]

        # Z de la caméra = Z du TCP + offset Z caméra (hauteur caméra au-dessus de la main)
        z_cam = robot_pose[2] + offsets[2]
        z_plan = z_target if z_target is not None else cfg.data["hauteurs"]["jeu"]

        # Distance caméra→plan de jeu en mètres (pour la déprojection)
        dist_m = (z_cam - z_plan) / 1000.0
        if dist_m <= 0: dist_m = 0.1  # Garde-fou : distance négative impossible physiquement

        # Déprojection : pixel (u,v) à distance dist_m → point 3D dans le repère caméra (en mètres)
        pt_cam = rs.rs2_deproject_pixel_to_point(self.intrinsics, [float(u), float(v)], float(dist_m))
        # Conversion mm + permutation axes caméra→robot
        cam_x, cam_y = pt_cam[0]*1000, pt_cam[1]*1000

        # La caméra regarde vers le bas : axe X caméra = axe Y robot, axe Y caméra = axe X robot
        vec_x_robot = cam_y
        vec_y_robot = cam_x

        pos_x = robot_pose[0] + offsets[0] + vec_x_robot
        pos_y = robot_pose[1] + offsets[1] + vec_y_robot
        return [pos_x, pos_y]

    def cleanup(self): #OK
        """Arrête proprement le pipeline RealSense."""
        if self.pipeline:
            try: self.pipeline.stop()
            except: pass

# ============================================================================
# 4. ASSISTANT DE CALIBRATION
# ============================================================================

def attendre_reglage_vision(vision): #OK
    """Interface interactive de réglage gamma/CLAHE via trackbars OpenCV.
    Les modifications sont appliquées en temps réel sur le flux caméra.
    ESPACE pour valider et sauvegarder."""
    print("\n[SETUP] ÉTAPE 0 : Réglage de la Vision")
    print("[VISUEL] Ajustez les curseurs pour bien voir les marqueurs ArUco.")
    print("[VISUEL] Appuyez sur ESPACE pour valider les réglages.")

    win_name = "Calibration Vision (Espace pour valider)"
    cv2.namedWindow(win_name)

    cv2.createTrackbar("Gamma (x10)", win_name, int(cfg.data["vision"]["gamma"] * 10), 50, lambda x: None)
    cv2.createTrackbar("Contraste", win_name, int(cfg.data["vision"]["clahe_clip"]), 20, lambda x: None)

    while True:
        g = cv2.getTrackbarPos("Gamma (x10)", win_name) / 10.0
        if g <= 0.1: g = 0.1  # Évite gamma=0 (image noire)
        c = cv2.getTrackbarPos("Contraste", win_name)

        # Mise à jour en temps réel des paramètres de vision (affectent capturer() immédiatement)
        cfg.data["vision"]["gamma"] = float(g)
        cfg.data["vision"]["clahe_clip"] = float(c)

        img_couleur, corners, ids = vision.capturer()
        if img_couleur is None:
            time.sleep(0.05)
            continue

        # Affiche l'image prétraitée (celle que voit le détecteur ArUco) pour aider au réglage
        img_robot = vision.get_processed_image(img_couleur)
        img_robot_color = cv2.cvtColor(img_robot, cv2.COLOR_GRAY2BGR)

        if ids is not None:
            aruco.drawDetectedMarkers(img_robot_color, corners, ids)
            cv2.putText(img_robot_color, _cv2txt(f"ArUco OK : {len(ids)} detectes"), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(img_robot_color, _cv2txt("Aucun ArUco visible"), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(img_robot_color, _cv2txt("[ESPACE] VALIDER"), (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(win_name, img_robot_color)

        if cv2.waitKey(1) == 32:  # Touche ESPACE
            print("[VISUEL] Réglages vision enregistrés.")
            break

    cv2.destroyWindow(win_name)

def attendre_validation_visuelle(vision, message_instruction, robot=None) -> bool: #OK
    """Boucle d'attente visuelle : affiche le flux caméra avec les marqueurs détectés.
    L'opérateur déplace manuellement le robot (via logiciel xArm externe) puis valide avec ESPACE.
    ESC pour annuler (demande confirmation).
    Retourne True si validé, False si annulé : le caller DOIT verifier et ne pas sauvegarder
    la pose courante en cas d'annulation (sinon la calibration se corrompt silencieusement)."""
    print(f"\n[VISUEL] {message_instruction}")
    if robot and not robot.simule:
        print("[VISUEL] Utilisez votre logiciel externe (xArm) pour déplacer le robot.")
    print("[VISUEL] Appuyez sur ESPACE pour valider, ECHAP pour annuler.")

    win_name = "Assistant Calibration (Espace pour valider)"
    cv2.namedWindow(win_name)

    valide = False
    try:
        while True:
            img, corners, ids = vision.capturer()

            if img is None:
                time.sleep(0.05)
                continue

            if ids is not None:
                aruco.drawDetectedMarkers(img, corners, ids)

            cv2.rectangle(img, (0, 0), (640, 60), (0, 0, 0), -1)  # Fond opaque pour le texte
            cv2.putText(img, _cv2txt(message_instruction[:50]), (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, _cv2txt("[ESPACE] Valider   [ECHAP] Annuler"), (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow(win_name, img)

            # & 0xFF pour eviter les codes >255 (touches speciales/meta) qui declenchent
            # des faux positifs d'annulation. waitKeyEx renvoyait parfois des bits hauts
            # venus d'autres fenetres et annulait la calib silencieusement.
            key = cv2.waitKey(1) & 0xFF

            if key == 32:  # ESPACE = validation
                print("[VISUEL] Position validée.")
                valide = True
                break
            if key == 27:  # ECHAP = annulation avec confirmation CLI (evite le faux positif)
                reponse = input("[VISUEL] Confirmer l'annulation de cette etape ? (o/n) ").strip().lower()
                if reponse == 'o':
                    print("[VISUEL] Annulation confirmee.")
                    valide = False
                    break
                print("[VISUEL] Annulation annulee, reprise.")
    finally:
        cv2.destroyWindow(win_name)
    return valide

def wizard_calibration(robot, vision, force=False): 
    """Assistant de calibration en 3 étapes :
      1. Réglage vision (gamma/CLAHE)
      2. Position SCAN (vue globale)
      3. Position STOCK + calibration correction mécanique et hauteur du plateau

    force=True : saute les questions o/n et force toutes les étapes (utilisé si config absente).

    DEBUG: si le robot manque toujours sa cible d'un biais constant, relancer uniquement l'étape 3.
    """
    print("\n[SETUP] --- ASSISTANT CALIBRATION INTERACTIVE ---")

    if vision.simule:
        print(" ATTENTION: Vision en mode SIMULATION. Calibration vision ignorée.")
    else:
        robot.retour_scan_securise()
        if force or input("[SETUP] Modifier les réglages de la Caméra (Lumière/Contraste) ? (o/n) ").lower() == 'o':
            attendre_reglage_vision(vision)
            cfg.sauvegarder()

    print("\n[SETUP] 1. Position SCAN (Vue globale)")
    robot.retour_scan_securise()
    if force or input("[SETUP] Modifier la position SCAN ? (o/n) ").lower() == 'o':
        if attendre_validation_visuelle(vision, "Ajustez la vue globale (SCAN)", robot):
            pos = robot.get_pose()
            cfg.data["pos_scan"] = [float(x) for x in pos]
            print(f"[SETUP] Position SCAN enregistrée : {cfg.data['pos_scan'][:3]}")
            cfg.sauvegarder()
        else:
            print("[SETUP] Etape SCAN annulee : position non modifiee.")
        robot.retour_scan_securise()
    else:
        print("[SETUP] Position SCAN conservée.")

    print("\n[SETUP] 2. Position STOCK (Vue d'en haut)")
    robot.retour_scan_securise()
    if force or input("[SETUP] Modifier la position STOCK (Vue d'en haut) ? (o/n) ").lower() == 'o':
        if not force:
            p_stock = cfg.data["pos_stock"]
            h_scan = cfg.data["pos_scan"][2]
            robot.deplacer(p_stock[0], p_stock[1], h_scan)
            robot.deplacer(p_stock[0], p_stock[1], p_stock[2])  # Descend à la hauteur stock

        if attendre_validation_visuelle(vision, "Centrez la caméra au-dessus du stock (en hauteur)", robot):
            pos = robot.get_pose()
            cfg.data["pos_stock"] = [float(x) for x in pos]
            print(f"[SETUP] Position STOCK enregistrée : {cfg.data['pos_stock'][:3]}")
            cfg.sauvegarder()
        else:
            print("[SETUP] Etape STOCK annulee : position non modifiee.")
        robot.retour_scan_securise()
    else:
        print("[SETUP] Position STOCK conservée.")

    print("\n[SETUP] 3. Calibration de la correction mécanique & hauteurs (Aruco 0)")
    robot.retour_scan_securise()
    if force or input("[SETUP] Lancer le calibrage automatique sur l'ArUco 0 ? (o/n) ").lower() == 'o':

        robot.retour_scan_securise()

        print("[SETUP] En attente de la stabilisation...")
        time.sleep(1.0)
        vision.purger_buffer(20)  # Frame frais après la stabilisation

        # Boucle multi-tentatives : l'ArUco 0 peut rater sur un seul frame (flou, exposition transitoire)
        # On tente jusqu'à MAX_ESSAIS captures en cherchant spécifiquement l'ID 0.
        MAX_ESSAIS = 8
        img, corners, ids = None, None, None
        for essai in range(MAX_ESSAIS):
            img, corners, ids = vision.capturer()
            if ids is not None and 0 in ids.flatten():
                break  # ID 0 trouvé → on arrête la boucle
            if essai < MAX_ESSAIS - 1:
                time.sleep(0.1)  # Courte pause pour laisser l'exposition se stabiliser

        if ids is not None:
            ids_flat = ids.flatten()
            # Préfère l'ArUco 0 (coin bas-gauche), sinon prend le premier marqueur visible
            target_id = 0 if 0 in ids_flat else ids_flat[0]

            if target_id in ids_flat:
                idx = np.where(ids_flat == target_id)[0][0]
                c = corners[idx][0]
                cx, cy = np.mean(c[:, 0]), np.mean(c[:, 1])

                pos_camera_photo = robot.get_pose()
                pos_rob_approx = vision.pixel_vers_robot(cx, cy, pos_camera_photo)

                if pos_rob_approx:
                    Z_POINTEUR = 180  # Hauteur intermédiaire de pointage (mm) . Robot pointe vers le marqueur
                    h_scan = cfg.data["pos_scan"][2]

                    # Déplace le robot au-dessus du marqueur (en 2 temps pour la sécurité)
                    robot.deplacer(pos_rob_approx[0], pos_rob_approx[1], h_scan)
                    robot.deplacer(pos_rob_approx[0], pos_rob_approx[1], Z_POINTEUR, speed=100)

                    time.sleep(1)
                    pos_initiale = robot.get_pose()[:3]

                    print("\n[VISUEL] Regardez le VRAI robot (pas l'écran) !")
                    instruction = "Descendez LA PINCE sur la base du plateau (Aruco 0)"
                    attendre_validation_visuelle(vision, instruction, robot)

                    time.sleep(0.5)  # Stabilisation après le mouvement manuel
                    pos_finale = robot.get_pose()[:3]
                    h_jeu_new = float(pos_finale[2])  # Z réel de la surface du plateau (mm)

                    # On recalcule la vraie cible théorique de projection X/Y maintenant qu'on a le VRAI Z
                    # (la déprojection dépend de la distance caméra→plan, qui change avec Z)
                    pos_parfaite = vision.pixel_vers_robot(cx, cy, pos_camera_photo, z_target=h_jeu_new)

                    # Le drift mécanique absolu est la différence entre là où le robot A DÛ aller et la théorie parfaite
                    # dx > 0 → robot arrive trop loin en X → correction négative nécessaire
                    dx = pos_finale[0] - pos_parfaite[0]
                    dy = pos_finale[1] - pos_parfaite[1]

                    print(f"\n[SETUP] Z Plateau calibré : {h_jeu_new:.1f}mm")
                    print(f"[SETUP] Jeu mécanique mesuré et corrigé : dX={dx:.1f}mm, dY={dy:.1f}mm")

                    # Remplacement (et non addition) de la correction !
                    # DEBUG: si on additionne, les biais s'accumulent à chaque calibration
                    cfg.data["correction_mecanique_xy"] = [float(dx), float(dy)]
                    cfg.data["hauteurs"]["jeu"] = h_jeu_new
                    # hauteur de vol = surface + 100mm (marge confortable au-dessus des pions)
                    cfg.data["hauteurs"]["vol"] = h_jeu_new + 100.0

                    robot.retour_scan_securise()
            else:
                print("[SETUP] ArUco 0 non trouvé.")

    # Sauvegarde finale forcée (même si aucune étape n'a été modifiée)
    cfg.sauvegarder()
    cfg.fichier_existe = True

# ============================================================================
# NOUVEAU: ASSISTANT D'ANALYSE PHYSIQUE DE TRICHE ET RE-VERIFICATION
# ============================================================================
def evaluer_etat_physique(ids, corners, cases_memory): #OK
    """Analyse l'image courante et détecte quel pion se trouve réellement sur chaque case.

    Paramètres :
      ids          : array (N,1) des IDs ArUco détectés, ou None
      corners      : liste de N arrays (1,4,2) des coins détectés correspondants
      cases_memory : {index_case → (cx,cy) pixel} . Positions de référence des cases

    Compare la position pixel de chaque marqueur pion avec les positions mémorisées des cases.
    Un pion est associé à la case la plus proche si la distance est < 60px (rayon de tolérance).

    Retourne phys_board : liste de 9 valeurs ('X', 'O', ou ' ') . État réellement observé par la caméra.

    DEBUG: si un pion est associé à la mauvaise case, ajuster le rayon best_dist (ligne `best_dist = 60`).
    """
    phys_board = [" "] * 9  # Résultat : 9 cases, initialement toutes vides
    if ids is None or corners is None:
        return phys_board

    ids_flat = ids.flatten()  # Array 1D des IDs détectés
    for cidx, case_pos in cases_memory.items():
        if case_pos is None:
            continue  # Case jamais vue par la caméra → on ne peut pas l'évaluer
        cx, cy = case_pos  # Centre pixel de référence de la case (mémorisé depuis la dernière détection ArUco)
        best_dist = 60  # Distance maximale (px) pour associer un pion à cette case . À ajuster si faux positifs/négatifs
        piece = " "     # Valeur par défaut : case vide

        for i, mid in enumerate(ids_flat):
            if mid in ID_PIONS_ROBOT or mid in ID_PIONS_HUMAIN:
                c = corners[i][0]
                pcx, pcy = int(np.mean(c[:, 0])), int(np.mean(c[:, 1]))
                dist = math.hypot(pcx - cx, pcy - cy)
                if dist < best_dist:
                    best_dist = dist
                    piece = "O" if mid in ID_PIONS_ROBOT else "X"
        phys_board[cidx] = piece
    return phys_board


def detecter_anomalies_plateau(board_ia: Plateau, phys_board: Plateau, case_ignore: int = -1):
    """Retourne les anomalies sous forme de tuples (case_idx, attendu, vu).
    - attendu in {"X", "O", " "}
    - vu in {"X", "O", " "}
    """
    anomalies = []
    for cidx in range(9):
        if cidx == case_ignore:
            continue
        attendu = board_ia[cidx]
        vu = phys_board[cidx]
        if attendu == " ":
            # Toute piece sur une case censee etre vide = incoherence (ex: 2e coup humain)
            if vu != " ":
                anomalies.append((cidx, attendu, vu))
            continue
        if vu != attendu:
            anomalies.append((cidx, attendu, vu))
    return anomalies


def contient_coup_supplementaire(anomalies) -> bool:
    """True si au moins une case attendue vide contient un pion.
    Cas typique: plusieurs coups joues dans le meme tour humain."""
    return any(attendu == " " and vu != " " for _, attendu, vu in anomalies)


def formater_message_reparation(anomalies) -> str:
    if not anomalies:
        return ""
    prefix = "1 seul coup autorise par tour. " if contient_coup_supplementaire(anomalies) else ""
    actions = []
    for cidx, attendu, vu in anomalies:
        case_txt = f"case {cidx + 1}"
        if attendu == " " and vu != " ":
            actions.append(f"{case_txt}: retirer {vu}")
        elif attendu != " " and vu == " ":
            actions.append(f"{case_txt}: remettre {attendu}")
        else:
            actions.append(f"{case_txt}: mettre {attendu} (pas {vu})")
    return prefix + "Corriger -> " + " ; ".join(actions)

# ============================================================================
# 5. BOUCLE PRINCIPALE DE JEU
# ============================================================================

def main():
    robot = RobotController(IP_ROBOT, cfg.data)  # Contrôleur robot . Tente la connexion au démarrage
    vision = VisionSystem()                       # Système vision . Tente d'initialiser la RealSense

    WIN_NAME = "MORPION BIP BOUP"  # Nom de la fenêtre OpenCV . Utilisé comme clé dans tous les cv2.imshow/waitKey
    cv2.namedWindow(WIN_NAME)
    vision.set_mouse_callback(WIN_NAME)  # Enregistre le handler clic sur cette fenêtre

    if not robot.simule:
        print("[MAIN] -> Mise en position SCAN initiale...")
        robot.retour_scan_securise()
        if not cfg.fichier_existe:
            print("\n" + "!"*50)
            print("[MAIN] CONFIGURATION MANQUANTE.")
            print("[MAIN] Lancement OBLIGATOIRE de l'assistant de calibration.")
            print("!"*50 + "\n")
            wizard_calibration(robot, vision, force=True)
        elif input("[MAIN] Lancer l'assistant de calibration ? (o/n) ").lower() == 'o':
            wizard_calibration(robot, vision, force=False)

    # Choix du niveau apres la calibration, modifiable a la volee via 1/2/3 en jeu
    niveau = choisir_difficulte_terminal()
    ia = MorpionIA(niveau)
    print(f"[MAIN] Niveau IA : {ia.difficulte}")

    print("\n[MAIN] === JEU PRET ===\n")

    # Zones de clic en mode simulation : grille 3x3 centrée, cases de 100x100px
    # Origin (ox, oy) = coin haut-gauche de la case 0 . Ajuster si l'image simulée est redimensionnée
    zones_clic = []   # Liste de tuples (x, y, largeur, hauteur) pour chaque case . Index = index de case
    ox, oy = 170, 90  # Origine de la grille (coin haut-gauche de la case 0) en pixels
    w, h = 100, 100   # Dimensions d'une case cliquable en pixels
    for i in range(3):
        for j in range(3):
            zones_clic.append((ox + j*w, oy + i*h, w, h))

    cases_memory = {i: None for i in range(9)}
    # cases_memory : dict {index_case → (cx, cy) pixel | None}
    # Mise à jour chaque frame depuis les marqueurs MAPPING_CASES (ID 10-18) détectés.
    # Sert de référence pixel persistante même quand un marqueur est momentanément caché.
    # None = case jamais vue depuis le démarrage (ArUco non détecté).

    presence_humain = {i: 0 for i in range(9)}
    # presence_humain : dict {index_case → compteur int}
    # Incrémenté chaque frame où un pion humain est détecté proche de la case.
    # Décrémenté si le pion s'éloigne. Un coup est validé à SEUIL_VALIDATION_HUMAIN.
    # Reset à 0 pour toutes les cases dès qu'un coup est validé.

    SEUIL_VALIDATION_HUMAIN = 30  # Nombre de frames consécutifs requis pour valider un coup humain
                                   # ≈ 1 seconde à 30fps . Évite les faux positifs de déplacement rapide
                                   # DEBUG: réduire si la validation est trop lente, augmenter si trop de faux positifs

    ids_sur_plateau = set()  # Set des IDs ArUco de pions posés sur le plateau à ce frame
                              # Passé comme ids_exclus à sequence_poser_pion() pour ne pas resaisir un pion déjà joué

    tour_robot = False   # False = attend le coup humain, True = le robot doit calculer et jouer son coup
    board_ia: Plateau = [" "] * 9  # Plateau logique . Source de verite pour la logique de jeu
                                    # Distinct de phys_board (etat physique observe) et current_display_board (affichage)

    # Detection de triche : au moment ou un coup humain est valide (apres SEUIL_VALIDATION_HUMAIN
    # frames stables), on compare phys_board au board_ia attendu. Une divergence sur une case
    # deja jouee signifie qu'un pion a ete retire ou remplace. Pour eviter les faux positifs
    # (main qui masque temporairement un pion), on n'alerte que si l'anomalie est confirmee
    # sur plusieurs verifications successives. Cette strategie ne declenche jamais la triche
    # pendant que l'humain pose son pion ou que le robot joue. Simple et robuste.
    is_triche_mode = False     # Flag actif tant que l'anomalie est visible dans le HUD
    msg_triche = ""            # Description courte de l'anomalie + consigne de correction
    anomalies_triche = []      # Liste des anomalies courantes (case_idx, attendu, vu)
    frames_ok_reparation = 0   # Nombre de frames consecutives sans anomalie
    SEUIL_CLEAR_TRICHE = 8     # Deblocage apres correction stable sur quelques frames
    msg_robot_actif = ""       # Message persistant affiché dans la barre HUD pendant tout le tour du robot
                               # Contient la case cible + coordonnées XYZ . Mis à jour dès que le coup est calculé

    try:
        while True: 
            img, corners, ids = vision.capturer()
            if img is None: break

            current_display_board = board_ia[:]  # Snapshot de board_ia pour ce frame . Mis à jour immédiatement
                                                  # après un coup humain pour un affichage réactif sans attendre le frame suivant
            ids_flat = ids.flatten() if ids is not None else []  # Array 1D des IDs ArUco détectés ce frame, ou liste vide

            # --- Mise à jour mémoire du plateau ---
            # Rafraîchit les positions pixel des cases ET détecte quels pions sont sur le plateau
            ids_sur_plateau.clear()  # Reconstruit à chaque frame
            if ids is not None:
                for i, mid in enumerate(ids_flat):
                    if mid in MAPPING_CASES:
                        # Marqueur de case (ID 10-18) → met à jour sa position dans cases_memory
                        cidx = MAPPING_CASES[mid]
                        c = corners[i][0]
                        cx, cy = int(np.mean(c[:, 0])), int(np.mean(c[:, 1]))
                        cases_memory[cidx] = (cx, cy)
                    else:
                        # Marqueur de pion → vérifie s'il est proche d'une case connue (= posé sur le plateau)
                        if mid in ID_PIONS_ROBOT or mid in ID_PIONS_HUMAIN:
                            c = corners[i][0]
                            cx, cy = int(np.mean(c[:, 0])), int(np.mean(c[:, 1]))
                            for cidx, pos in cases_memory.items():
                                if pos is not None and math.hypot(cx - pos[0], cy - pos[1]) < 80:
                                    # 80px = rayon de proximité "pion sur case" → pion considéré comme utilisé
                                    ids_sur_plateau.add(mid)
                                    break

            # En simulation sans ArUco : initialise les cases à des positions fixes dans l'image
            if vision.simule and all(v is None for v in cases_memory.values()):
                for idx, (zx, zy, zw, zh) in enumerate(zones_clic):
                    cases_memory[idx] = (int(zx + zw/2), int(zy + zh/2))

            # Surveillance continue de l'anomalie : permet de debloquer automatiquement
            # des que le plateau redevient coherent.
            if ACTIVER_DETECTION_TRICHE and is_triche_mode:
                phys_board_live = evaluer_etat_physique(ids, corners, cases_memory)
                anomalies_live = detecter_anomalies_plateau(board_ia, phys_board_live)
                if anomalies_live:
                    anomalies_triche = anomalies_live
                    msg_triche = formater_message_reparation(anomalies_live)
                    frames_ok_reparation = 0
                else:
                    frames_ok_reparation += 1
                    if frames_ok_reparation >= SEUIL_CLEAR_TRICHE:
                        is_triche_mode = False
                        msg_triche = ""
                        anomalies_triche = []
                        frames_ok_reparation = 0
                        print("[GAME] Plateau corrige. Reprise normale.")

            # --- Détection coup HUMAIN ---
            human_moved_idx = -1  # Index de la case jouée par l'humain ce frame, -1 si aucun coup détecté

            if not tour_robot and not (ACTIVER_DETECTION_TRICHE and is_triche_mode):
                if vision.simule and vision.click_coords:
                    # Mode simulation : coup humain via clic souris
                    mx, my = vision.click_coords
                    vision.click_coords = None
                    for idx, (zx, zy, zw, zh) in enumerate(zones_clic):
                        if zx < mx < zx+zw and zy < my < zy+zh and board_ia[idx] == " ":
                            human_moved_idx = idx

                elif ids is not None:
                    # Mode réel : coup humain détecté quand un pion ArUco humain est vu sur une case vide
                    # ET que le marqueur de la case est caché (= pion posé dessus)
                    cases_avec_pion_humain = []  # Liste des indices de cases où un pion humain est vu CE frame
                    for i, mid in enumerate(ids_flat):
                        if mid in ID_PIONS_HUMAIN:
                            c = corners[i][0]
                            cx, cy = int(np.mean(c[:, 0])), int(np.mean(c[:, 1]))
                            # cx, cy : centre pixel du pion humain dans l'image

                            best_cidx = -1   # Index de la case la plus proche de ce pion (-1 = aucune)
                            min_dist = 80    # Distance max (px) pour qu'un pion soit considéré "sur" une case
                            for cidx, pos in cases_memory.items():
                                if pos is not None and board_ia[cidx] == " ":
                                    id_case_plateau = CASE_TO_ARUCO.get(cidx)
                                    if id_case_plateau is None:
                                        continue
                                    if id_case_plateau not in ids_flat:  # Condition : marqueur de case caché = pion posé dessus
                                        dist = math.hypot(cx - pos[0], cy - pos[1])
                                        if dist < min_dist:
                                            min_dist = dist
                                            best_cidx = cidx

                            if best_cidx != -1:
                                cases_avec_pion_humain.append(best_cidx)

                    # Validation par accumulation : le coup n'est accepté qu'après SEUIL_VALIDATION_HUMAIN frames consécutifs
                    for cidx in range(9):
                        if cidx in cases_avec_pion_humain:
                            presence_humain[cidx] += 1
                            if presence_humain[cidx] >= SEUIL_VALIDATION_HUMAIN:
                                human_moved_idx = cidx
                                presence_humain = {k: 0 for k in range(9)}  # Reset tous les compteurs
                                break
                        else:
                            presence_humain[cidx] = max(0, presence_humain[cidx] - 1)  # Décroissance si le pion bouge

            # --- Validation Coup Humain ---
            if human_moved_idx != -1:
                if ACTIVER_DETECTION_TRICHE:
                    # Controle anti-triche : phys_board est stable (SEUIL_VALIDATION_HUMAIN
                    # frames viennent d'accumuler la presence). On verifie que les pions
                    # deja poses correspondent toujours au board_ia. Ne vaut que pour les
                    # cases ou on s'attend a voir un pion : une case attendue vide peut
                    # juste etre masquee par la main et ne declenche pas l'alerte.
                    phys_board = evaluer_etat_physique(ids, corners, cases_memory)
                    anomalies = detecter_anomalies_plateau(board_ia, phys_board, case_ignore=human_moved_idx)
                    if anomalies:
                        is_triche_mode = True
                        anomalies_triche = anomalies
                        msg_triche = formater_message_reparation(anomalies)
                        frames_ok_reparation = 0
                        details = "; ".join([
                            f"case {c+1} attendu {a if a != ' ' else 'rien'} vu {v if v != ' ' else 'rien'}"
                            for c, a, v in anomalies
                        ])
                        print(f"[GAME] /!\\ Incoherence plateau : {details}")
                        if contient_coup_supplementaire(anomalies):
                            print("[GAME] /!\\ 1 seul coup autorise par tour.")
                        print("[GAME] Corrigez les cases en rouge, puis replacez les bons pions.")
                    else:
                        is_triche_mode = False
                        msg_triche = ""
                        anomalies_triche = []
                        frames_ok_reparation = 0
                        print(f"[GAME] Humain joue case {human_moved_idx}.")
                        board_ia[human_moved_idx] = "X"
                        current_display_board[human_moved_idx] = "X"
                        tour_robot = True
                else:
                    print(f"[GAME] Humain joue case {human_moved_idx}.")
                    board_ia[human_moved_idx] = "X"
                    current_display_board[human_moved_idx] = "X"
                    tour_robot = True

            # --- Affichage de l'Interface (HUD) ---
            dessiner_hud(img, current_display_board, cases_memory, tour_robot, presence_humain, vision.simule,
                         message_robot=msg_robot_actif,
                         seuil_validation=SEUIL_VALIDATION_HUMAIN,
                         is_triche_mode=is_triche_mode,
                         msg_triche=msg_triche,
                         anomalies_triche=anomalies_triche,
                         difficulte=ia.difficulte)
            cv2.imshow(WIN_NAME, img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('1'): ia.set_difficulte("FACILE")
            elif key == ord('2'): ia.set_difficulte("MOYEN")
            elif key == ord('3'): ia.set_difficulte("DIFFICILE")

            # Avant le tour robot, le plateau doit etre strictement coherent avec board_ia.
            # Si un pion supplementaire apparait (ex: humain joue 2 coups), on bloque le round.
            if ACTIVER_DETECTION_TRICHE and tour_robot and not is_triche_mode:
                phys_board_robot = evaluer_etat_physique(ids, corners, cases_memory)
                anomalies_robot = detecter_anomalies_plateau(board_ia, phys_board_robot)
                if anomalies_robot:
                    is_triche_mode = True
                    anomalies_triche = anomalies_robot
                    msg_triche = formater_message_reparation(anomalies_robot)
                    frames_ok_reparation = 0
                    tour_robot = False
                    details = "; ".join([f"case {c+1} attendu {a if a != ' ' else 'rien'} vu {v if v != ' ' else 'rien'}" for c, a, v in anomalies_robot])
                    print(f"[GAME] /!\\ Plateau incoherent avant tour robot : {details}")
                    if contient_coup_supplementaire(anomalies_robot):
                        print("[GAME] /!\\ 1 seul coup autorise par tour.")
                    print("[GAME] Round bloque : corrigez les cases indiquees.")

            # --- VERIFICATION FIN DE PARTIE (après chaque coup) ---
            gagnant = None
            if verifier_gagnant(board_ia, "X"):
                print("\n[GAME] RESULTAT: HUMAIN GAGNANT.\n")
                robot.animation_defaite(vision, WIN_NAME)
                gagnant = "X"
            elif verifier_gagnant(board_ia, "O"):
                print("\n[GAME] RESULTAT: ROBOT GAGNANT.\n")
                robot.danse_victoire(vision, WIN_NAME)
                gagnant = "O"
            elif est_nul(board_ia):
                print("\n[GAME] RESULTAT: PARTIE NULLE.\n")
                robot.animation_nul(vision, WIN_NAME)
                gagnant = "NUL"

            if gagnant:
                print("[GAME] Fin de partie. Appuyez sur R pour recommencer (apres retrait des pions humains), Q pour quitter.")
                quitter_programme = False
                restart_requested = False

                # Boucle fin de partie interactive : on reste dans le meme programme
                # et on autorise une nouvelle partie sans relancer le script.
                while True:
                    img_fin, corners_fin, ids_fin = vision.capturer()
                    if img_fin is None:
                        img_fin = img.copy()

                    panel = img_fin.copy()
                    dessiner_fin_de_partie(panel, gagnant)

                    phys_fin = evaluer_etat_physique(ids_fin, corners_fin, cases_memory)
                    cases_humain = [idx + 1 for idx, val in enumerate(phys_fin) if val == "X"]
                    cases_robot = [idx + 1 for idx, val in enumerate(phys_fin) if val == "O"]

                    if cases_humain or cases_robot:
                        txt = "RETIREZ LES PIONS : " + ", ".join(str(c) for c in cases_humain) + " et " + ", ".join(str(c) for c in cases_robot)
                        col = C_ORANGE
                    else:
                        txt = "PLATEAU PRET - [R] NOUVELLE PARTIE  [Q] QUITTER"
                        col = C_VERT

                    cv2.rectangle(panel, (0, panel.shape[0] - 36), (panel.shape[1], panel.shape[0]), (12, 12, 16), -1)
                    cv2.putText(panel, _cv2txt(txt), (14, panel.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 2, cv2.LINE_AA)
                    cv2.imshow(WIN_NAME, panel)

                    k = cv2.waitKey(50) & 0xFF
                    if k == ord('q') or k == 27:
                        quitter_programme = True
                        break
                    if k == ord('r'):
                        if cases_humain:
                            print("[GAME] Retirez d'abord tous les pions humains (X) du plateau.")
                        else:
                            restart_requested = True
                            break

                    try:
                        if cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
                            quitter_programme = True
                            break
                    except Exception:
                        quitter_programme = True
                        break

                if quitter_programme:
                    break

                if restart_requested:
                    board_ia = [" "] * 9
                    presence_humain = {i: 0 for i in range(9)}
                    tour_robot = False

                    is_triche_mode = False
                    msg_triche = ""
                    anomalies_triche = []
                    frames_ok_reparation = 0

                    msg_robot_actif = ""
                    robot.last_sequence_alert = ""
                    robot.last_drop_info = None
                    vision.click_coords = None

                    print("[GAME] Nouvelle partie relancee.")
                    continue

            # --- TOUR ROBOT ---
            if tour_robot:
                coup = ia.meilleur_coup(board_ia)

                if coup != -1:
                    # Calcul de la position physique cible pour les logs et le HUD
                    h_jeu = cfg.data["hauteurs"]["jeu"]
                    pos_pix = cases_memory.get(coup)
                    if pos_pix:
                        if vision.simule:
                            x_mm, y_mm = _pixel_vers_xy_simu(int(pos_pix[0]), int(pos_pix[1]))
                        else:
                            xy = vision.pixel_vers_robot(pos_pix[0], pos_pix[1], robot.get_pose(), z_target=h_jeu)
                            x_mm, y_mm = int(round(xy[0])), int(round(xy[1]))
                        z_mm = int(round(h_jeu))
                        msg_robot_actif = f"Case {coup + 1} | X:{x_mm} Y:{y_mm} Z:{z_mm} mm"
                        print(f"[GAME] Robot vise case {coup + 1} -> X:{x_mm} mm  Y:{y_mm} mm  Z:{z_mm} mm")
                    else:
                        msg_robot_actif = f"Case {coup + 1} | position inconnue"
                        print(f"[GAME] Robot vise case {coup + 1} (position pixel non memorisee)")

                    reussite = robot.sequence_poser_pion(vision, board_ia, coup, cases_memory, WIN_NAME, ids_sur_plateau)
                    if reussite:
                        board_ia[coup] = "O"
                        tour_robot = False
                        msg_robot_actif = ""
                        print(f"[GAME] Pion posé en case {coup}.")
                    else:
                        if robot.last_sequence_alert:
                            msg_robot_actif = robot.last_sequence_alert
                            print(f"[GAME] {robot.last_sequence_alert}")
                        else:
                            msg_robot_actif = "ALERTE: sequence robot avortee"
                            print("[GAME] Séquence avortée (limites dépassées ou erreur matérielle).")
                        # On réessaiera la boucle d'après
                else:
                    # Aucun coup possible (ne devrait pas arriver si est_nul() est vérifié avant)
                    break

    finally:
        # Nettoyage garanti même en cas d'exception ou de break
        vision.cleanup()
        if robot.arm and not robot.simule:
            try: robot.arm.disconnect()
            except: pass
        cv2.destroyAllWindows()
        print("\n[MAIN] === PROGRAMME TERMINE ===\n")

if __name__ == "__main__":
    main()
