from hashlib import md5


def generate_key(patient_id: str, study_id: str, series_id: str):
    return md5(f"{patient_id}+{study_id}+{series_id}".encode('utf-8')).hexdigest()
