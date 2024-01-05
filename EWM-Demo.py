import numpy as np
import pandas as pd

def entropy_weight_method_detailed(data):
    # Verileri DataFrame'e dönüştür ve transpoze al
    decision_matrix = pd.DataFrame(data).T
    print("Adım 1: Karar Matrisi Oluşturma")
    print(decision_matrix, "\n")

    # Sütunların toplamını hesapla
    column_sums = decision_matrix.sum(axis=0)
    print("Adım 2: Sütun Değerlerini Toplama")
    print("Sütun Toplamları:", column_sums, "\n")

    # Normalize matrisi hesapla
    normalized_matrix = decision_matrix.div(column_sums, axis=1)
    print("Adım 3: Normalize Matrisi Hesaplama")
    print("Normalize Matrisi:\n", normalized_matrix, "\n")

    # DEVIATION SEQUENCES matrisini hesapla
    deviation_sequences = normalized_matrix * np.log(normalized_matrix + np.finfo(float).eps)
    print("Adım 4: DEVIATION SEQUENCES Matrisi Hesaplama")
    print("DEVIATION SEQUENCES Matrisi:\n", deviation_sequences, "\n")

    # DEVIATION SEQUENCES sütun toplamlarını hesapla
    deviation_sums = deviation_sequences.sum(axis=0)
    print("Adım 5: DEVIATION SEQUENCES Sütun Toplamlarını Hesaplama")
    print("DEVIATION SEQUENCES Sütun Toplamları:", deviation_sums, "\n")

    # Ej değerlerini hesapla
    Ej = -(1 / np.log(decision_matrix.shape[0])) * deviation_sums
    print("Adım 6: Ej Değerlerini Hesaplama")
    print("Ej Değerleri:", Ej, "\n")

    # (1-Ej) değerlerini hesapla
    one_minus_Ej = 1 - Ej
    print("Adım 7: (1-Ej) Değerlerini Hesaplama")
    print("(1-Ej) Değerleri:", one_minus_Ej, "\n")

    # Sum_(1-Ej) değerini hesapla
    sum_one_minus_Ej = np.sum(one_minus_Ej)
    print("Adım 8: Sum_(1-Ej) Değerini Hesaplama")
    print("Sum_(1-Ej):", sum_one_minus_Ej, "\n")

    # W ağırlık matrisini hesapla
    W = one_minus_Ej / sum_one_minus_Ej
    print("Adım 9: W Ağırlık Matrisini Hesaplama")
    print("W Ağırlık Matrisi:\n", W, "\n")

    return W

# Örnek veri seti
data = {
    'Kriter1': [1, 2, 2, 3],
    'Kriter2': [1/2, 1, 2, 2],
    'Kriter3': [1/2, 1/2, 1, 2],
    'Kriter4': [1/3, 1/2, 1/2, 1]
}

# Detaylı Entropi Ağırlık Yöntemi fonksiyonunu çağır
weights = entropy_weight_method_detailed(data)

