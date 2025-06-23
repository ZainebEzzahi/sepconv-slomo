#!/usr/bin/env python

import getopt # Gardé pour la compatibilité avec le code original, bien que argparse prenne le relais
import math
import numpy
import PIL
import PIL.Image
import sys
import torch
import moviepy.video.io.ffmpeg_writer as ffmpeg_writer
from moviepy.editor import VideoFileClip

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
import argparse
import os

# Import du module sepconv personnalisé
import sepconv # the custom separable convolution layer

##########################################################

# S'assurer que les gradients ne sont pas calculés pour des raisons de performance
torch.set_grad_enabled(False) 

# Utiliser cuDNN pour des performances de calcul optimisées si disponible
torch.backends.cudnn.enabled = True 

##########################################################

# Définition de la classe Network (architecture du modèle SepConv)
class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Couches de base pour le réseau convolutif
        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False)
            )
        # end

        # Couches d'upsampling
        def Upsample(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False)
            )
        # end

        # Sous-réseau pour générer les noyaux de convolution séparables
        def Subnet():
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=51, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=51, out_channels=51, kernel_size=3, stride=1, padding=1)
            )
        # end

        # Définition des différentes couches du réseau
        self.netConv1 = Basic(6, 32)
        self.netConv2 = Basic(32, 64)
        self.netConv3 = Basic(64, 128)
        self.netConv4 = Basic(128, 256)
        self.netConv5 = Basic(256, 512)

        self.netDeconv5 = Basic(512, 512)
        self.netDeconv4 = Basic(512, 256)
        self.netDeconv3 = Basic(256, 128)
        self.netDeconv2 = Basic(128, 64)

        self.netUpsample5 = Upsample(512, 512)
        self.netUpsample4 = Upsample(256, 256)
        self.netUpsample3 = Upsample(128, 128)
        self.netUpsample2 = Upsample(64, 64)

        self.netVertical1 = Subnet()
        self.netVertical2 = Subnet()
        self.netHorizontal1 = Subnet()
        self.netHorizontal2 = Subnet()

        # Chargement des poids du modèle pré-entraîné
        # Utilise args_strModel qui sera défini par argparse
        self.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.hub.load_state_dict_from_url(url='http://content.sniklaus.com/sepconv/network-' + args_strModel + '.pytorch', file_name='sepconv-' + args_strModel).items() })
    # end

    # Méthode forward pour le passage avant du réseau
    def forward(self, tenOne, tenTwo):
        tenConv1 = self.netConv1(torch.cat([tenOne, tenTwo], 1))
        tenConv2 = self.netConv2(torch.nn.functional.avg_pool2d(input=tenConv1, kernel_size=2, stride=2, count_include_pad=False))
        tenConv3 = self.netConv3(torch.nn.functional.avg_pool2d(input=tenConv2, kernel_size=2, stride=2, count_include_pad=False))
        tenConv4 = self.netConv4(torch.nn.functional.avg_pool2d(input=tenConv3, kernel_size=2, stride=2, count_include_pad=False))
        tenConv5 = self.netConv5(torch.nn.functional.avg_pool2d(input=tenConv4, kernel_size=2, stride=2, count_include_pad=False))

        tenDeconv5 = self.netUpsample5(self.netDeconv5(torch.nn.functional.avg_pool2d(input=tenConv5, kernel_size=2, stride=2, count_include_pad=False)))
        tenDeconv4 = self.netUpsample4(self.netDeconv4(tenDeconv5 + tenConv5))
        tenDeconv3 = self.netUpsample3(self.netDeconv3(tenDeconv4 + tenConv4))
        tenDeconv2 = self.netUpsample2(self.netDeconv2(tenDeconv3 + tenConv3))

        tenCombine = tenDeconv2 + tenConv2

        tenVerone = self.netVertical1(tenCombine)
        tenVertwo = self.netVertical2(tenCombine)
        tenHorone = self.netHorizontal1(tenCombine)
        tenHortwo = self.netHorizontal2(tenCombine)

        return sum([
            sepconv.sepconv_func.apply(torch.nn.functional.pad(input=tenOne, pad=[int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0))], mode='replicate'), tenVerone, tenHorone),
            sepconv.sepconv_func.apply(torch.nn.functional.pad(input=tenTwo, pad=[int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0))], mode='replicate'), tenVertwo, tenHortwo)
        ])
    # end
# end

netNetwork = None # Variable globale pour stocker le réseau chargé

##########################################################

# Fonction d'estimation qui utilise le réseau pour interpoler une frame
def estimate(tenOne, tenTwo):
    global netNetwork

    if netNetwork is None:
        # Initialisation du réseau sur GPU
        netNetwork = Network().cuda().train(False)
    # end

    assert(tenOne.shape[1] == tenTwo.shape[1])
    assert(tenOne.shape[2] == tenTwo.shape[2])

    intWidth = tenOne.shape[2]
    intHeight = tenOne.shape[1]

    # Avertissements sur la taille des images pour éviter des problèmes de mémoire ou de performance
    assert(intWidth <= 1280) 
    assert(intHeight <= 720) 

    # Pré-traitement des frames (envoi au GPU et redimensionnement)
    tenPreprocessedOne = tenOne.cuda().view(1, 3, intHeight, intWidth)
    tenPreprocessedTwo = tenTwo.cuda().view(1, 3, intHeight, intWidth)

    # Définition du padding selon le mode choisi (paper ou improved)
    if args_strPadding == 'paper':
        intPaddingLeft, intPaddingTop, intPaddingBottom, intPaddingRight = int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)) ,int(math.floor(51 / 2.0))

    elif args_strPadding == 'improved':
        intPaddingLeft, intPaddingTop, intPaddingBottom, intPaddingRight = 0, 0, 0, 0
    # end

    intPreprocessedWidth = intPaddingLeft + intWidth + intPaddingRight
    intPreprocessedHeight = intPaddingTop + intHeight + intPaddingBottom

    # Ajustement des dimensions pour qu'elles soient des multiples de 128 (requis par le modèle)
    if intPreprocessedWidth != ((intPreprocessedWidth >> 7) << 7):
        intPreprocessedWidth = (((intPreprocessedWidth >> 7) + 1) << 7) 
    # end
    
    if intPreprocessedHeight != ((intPreprocessedHeight >> 7) << 7):
        intPreprocessedHeight = (((intPreprocessedHeight >> 7) + 1) << 7) 
    # end

    intPaddingRight = intPreprocessedWidth - intWidth - intPaddingLeft
    intPaddingBottom = intPreprocessedHeight - intHeight - intPaddingTop

    # Application du padding et passage des frames au réseau
    tenPreprocessedOne = torch.nn.functional.pad(input=tenPreprocessedOne, pad=[intPaddingLeft, intPaddingRight, intPaddingTop, intPaddingBottom], mode='replicate')
    tenPreprocessedTwo = torch.nn.functional.pad(input=tenPreprocessedTwo, pad=[intPaddingLeft, intPaddingRight, intPaddingTop, intPaddingBottom], mode='replicate')

    # Retourne la frame interpolée, recadrée et déplacée vers le CPU
    return torch.nn.functional.pad(input=netNetwork(tenPreprocessedOne, tenPreprocessedTwo), pad=[0 - intPaddingLeft, 0 - intPaddingRight, 0 - intPaddingTop, 0 - intPaddingBottom], mode='replicate')[0, :, :, :].cpu()
# end

##########################################################

# --- Fonctions utilitaires pour le chargement et l'interpolation linéaire ---

# Fonction pour charger une vidéo et la convertir en une liste de frames NumPy
def load_video_frames(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erreur : Impossible d'ouvrir la vidéo {video_path}")
        return None
    while True:
        ret, frame = cap.read() # Lire frame par frame
        if not ret:
            break
        frames.append(frame) # Stocker les frames en format NumPy (BGR par défaut)
    cap.release()
    return frames

# Fonction pour créer des frames interpolées linéaires entre deux frames (NumPy)
def interpolate_linear(frame1, frame2, num_interpolated_frames):
    interpolated_list = []
    for i in range(1, num_interpolated_frames + 1):
        alpha = i / (num_interpolated_frames + 1)
        # Assurez-vous que les types de données sont floats pour le calcul
        interpolated_frame = cv2.addWeighted(frame1.astype(np.float32), 1 - alpha, frame2.astype(np.float32), alpha, 0)
        interpolated_list.append(interpolated_frame.astype(np.uint8)) # Reconvertir en uint8 pour la sauvegarde
    return interpolated_list

# Fonction récursive pour générer les interpolations SepConv
# sf = 2^k, où k est le niveau de récursion.
# Par exemple, sf=2 (k=1) -> 1 frame intermédiaire
# sf=4 (k=2) -> 3 frames intermédiaires
# sf=8 (k=3) -> 7 frames intermédiaires
def get_sepconv_interpolations_recursive(frame_a_torch, frame_b_torch, current_level, max_level):
    # Condition d'arrêt de la récursion
    if current_level == max_level:
        return []
    
    # Estimer la frame du milieu (première interpolation à ce niveau)
    interp_mid_torch = estimate(frame_a_torch, frame_b_torch)
    
    # Appels récursifs pour les moitiées gauche et droite de l'intervalle
    left_interps = get_sepconv_interpolations_recursive(frame_a_torch, interp_mid_torch, current_level + 1, max_level)
    right_interps = get_sepconv_interpolations_recursive(interp_mid_torch, frame_b_torch, current_level + 1, max_level)
    
    # Combiner les résultats : interpolations gauche + frame du milieu actuelle + interpolations droite
    return left_interps + [interp_mid_torch] + right_interps


if __name__ == '__main__':
    # Unified argparse for all command-line arguments
    parser = argparse.ArgumentParser(description='SepConv for video slow-motion with PSNR/SSIM evaluation.')
    parser.add_argument('--model', type=str, default='l1', help='Model to use (l1 or lf).')
    parser.add_argument('--padding', type=str, default='improved', help='Which padding to use: "paper" or "improved".')
    parser.add_argument('--video', type=str, required=True, help='Path to the input video.')
    parser.add_argument('--out', type=str, required=True, help='Path to where the output slow-motion video should be stored.')
    parser.add_argument('--sf', type=int, default=4, help='Slow motion factor (e.g., 2 for 1 added frame, 4 for 3 added frames, etc.). Must be a power of 2.')
    parser.add_argument('--save_frames', action='store_true', help='Set this flag to save individual interpolated frames and their linear references.')
    parser.add_argument('--output_dir', type=str, default='evaluation_frames', help='Directory to save individual frames if --save_frames is set.')
    
    # Placeholder for image-based arguments (not used in video mode but kept for compatibility with original script)
    parser.add_argument('--one', type=str, default='./images/one.png', help='Path to the first frame (for image mode).')
    parser.add_argument('--two', type=str, default='./images/two.png', help='Path to the second frame (for image mode).')

    args = parser.parse_args()

    # Définition des variables globales utilisées par le modèle SepConv
    args_strPadding = args.padding
    args_strModel = args.model

    # Vérification du facteur de ralenti (sf)
    # math.log2(args.sf) doit être un entier pour que sf soit une puissance de 2
    if not (args.sf > 0 and (args.sf & (args.sf - 1) == 0)): # Check if sf is a power of 2
        print(f"Erreur: Le facteur de ralenti (sf) doit être une puissance de 2 (e.g., 2, 4, 8). Vous avez fourni {args.sf}.")
        sys.exit(1)

    # --- Logique de traitement vidéo ---
    if args.out.split('.')[-1] in ['avi', 'mp4', 'webm', 'wmv']:
        objVideoreader = VideoFileClip(filename=args.video)

        intWidth = objVideoreader.w # Correction ici de objVideorere.w en objVideoreader.w
        intHeight = objVideoreader.h
        original_fps = objVideoreader.fps

        # Chargement de toutes les frames originales en mémoire
        print(f"Chargement des frames de la vidéo originale : '{args.video}'...")
        original_frames = [np.array(frame) for frame in objVideoreader.iter_frames()]
        objVideoreader.close() # Fermer le VideoFileClip après le chargement

        if not original_frames:
            print(f"Erreur : Aucune frame chargée de la vidéo '{args.video}'. Vérifiez le chemin ou le format.")
            sys.exit(1)

        # Initialisation de l'enregistreur vidéo pour la sortie ralentie
        # *** CORRECTION MAJEURE ICI : Définir le FPS de sortie au FPS original pour un VRAI ralenti ***
        output_fps = original_fps 
        print(f"Initialisation de l'enregistreur vidéo pour '{args.out}' à {output_fps} FPS (pour un ralenti {args.sf}x) ...")
        objVideowriter = ffmpeg_writer.FFMPEG_VideoWriter(filename=args.out, size=(intWidth, intHeight), fps=output_fps)

        # Initialisation des listes pour les métriques
        psnr_scores_interpolated = []
        ssim_scores_interpolated = []
        frame_metrics_details = []
        global_interpolated_frame_idx = 0 # Compteur global pour les frames interpolées

        # Création des répertoires de sortie si la sauvegarde des frames est activée
        if args.save_frames:
            model_output_dir = os.path.join(args.output_dir, 'model_interpolated_frames')
            linear_reference_dir = os.path.join(args.output_dir, 'linear_reference_frames')
            os.makedirs(model_output_dir, exist_ok=True)
            os.makedirs(linear_reference_dir, exist_ok=True)
            print(f"Les frames interpolées et de référence seront sauvegardées dans '{args.output_dir}'")

        print("\n--- Démarrage de la génération vidéo et du calcul des métriques ---")

        # Écriture de la toute première frame originale (elle n'est pas interpolée)
        if original_frames:
            objVideowriter.write_frame(original_frames[0].astype(numpy.uint8)) 

        # Boucle sur les paires de frames originales
        for i in range(len(original_frames) - 1):
            frame_A_np = original_frames[i]
            frame_B_np = original_frames[i+1]

            # Conversion des frames NumPy en tenseurs Torch pour SepConv
            frame_A_torch = torch.FloatTensor(numpy.ascontiguousarray(frame_A_np[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
            frame_B_torch = torch.FloatTensor(numpy.ascontiguousarray(frame_B_np[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))

            # --- Génération des frames interpolées par SepConv ---
            # Nombre de frames intermédiaires à générer
            num_inserted_frames = args.sf - 1
            # Le niveau de récursion max_level pour un sf = 2^k est k
            max_recursion_level = int(math.log2(args.sf))

            # Appel de la fonction récursive pour obtenir les frames interpolées du modèle
            model_interpolated_torch_list = get_sepconv_interpolations_recursive(frame_A_torch, frame_B_torch, 0, max_recursion_level)
            
            # Génération des frames de "pseudo-ground truth" linéaires (NumPy)
            linear_interpolated_np_list = interpolate_linear(frame_A_np, frame_B_np, num_inserted_frames)

            # --- Traitement de chaque frame interpolée (métriques, sauvegarde, écriture) ---
            for j in range(len(model_interpolated_torch_list)):
                model_frame_torch = model_interpolated_torch_list[j]
                linear_frame_np = linear_interpolated_np_list[j]

                # Conversion de la frame du modèle de Torch à NumPy pour les métriques et la sauvegarde
                model_frame_np = (model_frame_torch.clip(0.0, 1.0).numpy(force=True).transpose(1, 2, 0)[:, :, ::-1] * 255.0).astype(numpy.uint8)

                # Assurer que les dimensions correspondent pour le calcul des métriques
                if model_frame_np.shape != linear_frame_np.shape:
                    linear_frame_np = cv2.resize(linear_frame_np, (model_frame_np.shape[1], model_frame_np.shape[0]), interpolation=cv2.INTER_AREA)

                # Convertir en niveaux de gris pour le calcul du PSNR/SSIM
                model_frame_gray = cv2.cvtColor(model_frame_np, cv2.COLOR_BGR2GRAY)
                linear_frame_gray = cv2.cvtColor(linear_frame_np, cv2.COLOR_BGR2GRAY)

                current_psnr = psnr_metric(linear_frame_gray, model_frame_gray, data_range=255)
                current_ssim = ssim_metric(linear_frame_gray, model_frame_gray, data_range=255)

                psnr_scores_interpolated.append(current_psnr)
                ssim_scores_interpolated.append(current_ssim)

                # Stocker les détails pour cette frame
                frame_metrics_details.append({
                    'original_interval_idx': i,
                    'interpolated_frame_pos_in_interval': j,
                    'global_interpolated_frame_idx': global_interpolated_frame_idx,
                    'psnr': current_psnr,
                    'ssim': current_ssim
                })

                # Afficher les métriques progressivement
                print(f"Interval {i:04d} / Frame {j:02d} (Global {global_interpolated_frame_idx:06d}): PSNR={current_psnr:.2f} dB, SSIM={current_ssim:.4f}")

                # Sauvegarder les frames si le flag est activé
                if args.save_frames:
                    filename_prefix = f"int{i:04d}_pos{j:02d}_global{global_interpolated_frame_idx:06d}"
                    cv2.imwrite(os.path.join(model_output_dir, f"{filename_prefix}_model_psnr{current_psnr:.2f}_ssim{current_ssim:.4f}.png"), model_frame_np)
                    cv2.imwrite(os.path.join(linear_reference_dir, f"{filename_prefix}_linear.png"), linear_frame_np)

                # Écrire la frame générée dans la vidéo de sortie
                objVideowriter.write_frame(model_frame_np)
                global_interpolated_frame_idx += 1

            # Après toutes les interpolations pour cet intervalle, écrire la frame originale B
            # Cette frame B deviendra la frame A pour l'intervalle suivant
            objVideowriter.write_frame(frame_B_np.astype(numpy.uint8))
        
        objVideowriter.close()
        print(f"\n--- Vidéo ralentie '{args.out}' générée avec succès. ---")

        # --- Résumé final des métriques ---
        if psnr_scores_interpolated and ssim_scores_interpolated:
            avg_psnr = np.mean(psnr_scores_interpolated)
            avg_ssim = np.mean(ssim_scores_interpolated)

            print(f"\n--- Résultats Globaux sur les frames interpolées ---")
            print(f"Moyenne PSNR : {avg_psnr:.2f} dB")
            print(f"Moyenne SSIM : {avg_ssim:.4f}")

            # Optionnel: afficher tous les détails à la fin si tu le souhaites (peut être très long)
            # print(f"\n--- Tous les Détails par Frame Interpolée ---")
            # for detail in frame_metrics_details:
            #     print(f"Global Interpolated Frame {detail['global_interpolated_frame_idx']} (Interval {detail['original_interval_idx']}, Pos {detail['interpolated_frame_pos_in_interval']}): PSNR={detail['psnr']:.2f} dB, SSIM={detail['ssim']:.4f}")

        else:
            print("Aucune frame interpolée à comparer pour les résultats globaux.")

    # --- Logique originale pour le mode image unique (si l'output n'est pas une vidéo) ---
    elif args.out.split('.')[-1] in ['bmp', 'jpg', 'jpeg', 'png']:
        print("Mode image unique détecté. Pas de calcul PSNR/SSIM pour ce mode, qui est conçu pour l'interpolation de vidéo.")
        tenOne = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(PIL.Image.open(args.one))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
        tenTwo = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(PIL.Image.open(args.two))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))

        tenOutput = estimate(tenOne, tenTwo)

        PIL.Image.fromarray((tenOutput.clip(0.0, 1.0).numpy(force=True).transpose(1, 2, 0)[:, :, ::-1] * 255.0).astype(numpy.uint8)).save(args.out)
        print(f"Image interpolée '{args.out}' générée avec succès.")
