#!/usr/bin/env python3
"""
Generate a small synthetic Echocardiography dataset (CSV) with common measurements.
Target size: < 1 MB
"""

import pandas as pd
import numpy as np
from pathlib import Path


np.random.seed(42)

NUM_PATIENTS = 200

# Typical echo value ranges (approximate)
RANGES = {
    'age': (25, 85),
    'sex': (0, 1),  # 0=F, 1=M
    'height_cm': (150, 200),
    'weight_kg': (45, 120),
    'bsa_m2': (1.4, 2.5),
    'heart_rate_bpm': (45, 120),
    'lvef_percent': (25, 75),          # left ventricular ejection fraction
    'lv_edv_ml': (60, 220),            # end-diastolic volume
    'lv_esv_ml': (20, 160),            # end-systolic volume
    'lv_mass_g': (80, 300),
    'lv_mass_index_g_m2': (40, 170),
    'lvidd_mm': (35, 65),              # LV internal diameter diastole
    'lvids_mm': (20, 45),              # LV internal diameter systole
    'ivsd_mm': (6, 14),                # interventricular septum diastole
    'pwed_mm': (6, 14),                # posterior wall thickness diastole
    'rv_diameter_mm': (20, 45),
    'la_diameter_mm': (25, 50),
    'ra_area_cm2': (8, 25),
    'tapsem_mm': (12, 28),             # tricuspid annular plane systolic excursion
    'e_ae_ratio': (0.5, 2.5),
    'e_wave_m_s': (0.4, 1.2),
    'a_wave_m_s': (0.3, 1.0),
    'dt_ms': (120, 280),               # deceleration time
    's_prime_cm_s': (5, 15),
    'e_prime_cm_s': (4, 14),
    'e_over_e_prime': (5, 20),
    'pulm_art_pressure_mmHg': (15, 55),
    'aortic_root_mm': (25, 40),
    'ascending_aorta_mm': (25, 42),
    'mitral_regurg_grade': (0, 4),
    'aortic_regurg_grade': (0, 4),
    'tricuspid_regurg_grade': (0, 4),
    'rvsp_mmHg': (15, 60)
}


def generate_patient(patient_index: int) -> dict:
    values = {key: np.round(np.random.uniform(low, high), 2) for key, (low, high) in RANGES.items()}
    values['patient_id'] = f"ECHO_{patient_index:04d}"

    # sex as integer 0/1
    values['sex'] = int(np.random.binomial(1, 0.5))

    # Body surface area via DuBois formula (approx.; using random ht/wt to avoid circular dependency)
    values['bsa_m2'] = round(0.007184 * (np.random.uniform(150, 200)) ** 0.725 * (np.random.uniform(45, 120)) ** 0.425, 2)

    # Ensure EDV > ESV
    if values['lv_esv_ml'] >= values['lv_edv_ml']:
        values['lv_esv_ml'] = round(max(values['lv_edv_ml'] - np.random.uniform(10, 60), 10), 2)

    # LVEF from volumes with slight noise
    lvef_calc = max(0.01, min(0.85, (values['lv_edv_ml'] - values['lv_esv_ml']) / values['lv_edv_ml']))
    values['lvef_percent'] = round(100 * (lvef_calc + np.random.normal(0, 0.02)), 1)

    # Regurgitation grades as integers
    for grade_key in ['mitral_regurg_grade', 'aortic_regurg_grade', 'tricuspid_regurg_grade']:
        values[grade_key] = int(np.random.choice([0, 1, 2, 3, 4], p=[0.6, 0.2, 0.12, 0.06, 0.02]))

    # Derived clinical labels
    values['label_hf_ref'] = int(values['lvef_percent'] < 40)
    lvmi_threshold = 115 if values['sex'] == 1 else 95
    values['label_lv_hypertrophy'] = int(values['lv_mass_index_g_m2'] > lvmi_threshold)

    return values


def main() -> None:
    rows = [generate_patient(i) for i in range(1, NUM_PATIENTS + 1)]
    df = pd.DataFrame(rows)

    datasets_dir = Path('datasets')
    datasets_dir.mkdir(exist_ok=True)
    out_file = datasets_dir / 'echo_sample.csv'
    df.to_csv(out_file, index=False)
    print(f"✅ Wrote {out_file} ({out_file.stat().st_size/1024:.1f} KB, {len(df)} patients)")

    # Copy to webassembly sample_data
    web_dir = Path('webassembly/sample_data')
    web_dir.mkdir(exist_ok=True)
    web_file = web_dir / 'test_echo_sample.csv'
    df.to_csv(web_file, index=False)
    print(f"✅ Copied to {web_file}")


if __name__ == '__main__':
    main()





