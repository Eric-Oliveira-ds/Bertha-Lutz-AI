import { onlyDigits } from "./masks";

const MIN_NAME_LENGTH = 3;
const MIN_BIRTH_YEAR = 1900;

export function validarNome(nome) {
  return nome.trim().length >= MIN_NAME_LENGTH;
}

export function validarCPF(cpf) {
  const digits = onlyDigits(cpf);

  if (digits.length !== 11 || /^(\d)\1+$/.test(digits)) {
    return false;
  }

  const isValidDigit = (length) => {
    let sum = 0;
    for (let i = 0; i < length; i += 1) {
      sum += parseInt(digits[i], 10) * (length + 1 - i);
    }
    const rest = (sum * 10) % 11;
    return rest === 10 || rest === 11 ? 0 : rest;
  };

  return (
    isValidDigit(9) === parseInt(digits[9], 10) &&
    isValidDigit(10) === parseInt(digits[10], 10)
  );
}

export function validarDataNascimento(data) {
  const match = /^(\d{2})\/(\d{2})\/(\d{4})$/.exec(data);
  if (!match) return false;

  const day = parseInt(match[1], 10);
  const month = parseInt(match[2], 10);
  const year = parseInt(match[3], 10);

  if (month < 1 || month > 12) return false;
  if (year < MIN_BIRTH_YEAR || year > new Date().getFullYear()) return false;

  const daysInMonth = new Date(year, month, 0).getDate();
  return day >= 1 && day <= daysInMonth;
}

export function validarTelefone(telefone) {
  const digits = onlyDigits(telefone);
  return digits.length >= 10 && digits.length <= 11;
}

export function validarFormulario({ name, cpf, dateBirth, phone }) {
  if (!validarNome(name)) return "Nome deve ter no mínimo 3 caracteres";
  if (!validarCPF(cpf)) return "CPF deve conter 11 dígitos válidos";
  if (!validarDataNascimento(dateBirth)) return "Data de nascimento inválida (DD/MM/AAAA)";
  if (!validarTelefone(phone)) return "Telefone inválido — informe DDD + número";
  return null;
}