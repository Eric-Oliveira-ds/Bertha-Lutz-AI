from fastapi import HTTPException


def validar_cpf(cpf: str):
    cpf_limpo = "".join(filter(str.isdigit, cpf))

    if len(cpf_limpo) != 11:
        raise HTTPException(
            status_code=400,
            detail="CPF deve conter 11 dígitos"
        )