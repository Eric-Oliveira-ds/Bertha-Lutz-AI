import { useState } from "react";
import axios from "axios";

function validarCPF(cpf) {

  cpf = cpf.replace(/[^\d]+/g,'');

  if (cpf.length !== 11 || /^(\d)\1+$/.test(cpf))
    return false;

  let soma = 0;
  let resto;

  for (let i=1; i<=9; i++)
    soma = soma + parseInt(cpf.substring(i-1, i)) * (11 - i);

  resto = (soma * 10) % 11;

  if ((resto == 10) || (resto == 11))
    resto = 0;

  if (resto != parseInt(cpf.substring(9, 10)))
    return false;

  soma = 0;

  for (let i = 1; i <= 10; i++)
    soma = soma + parseInt(cpf.substring(i-1, i)) * (12 - i);

  resto = (soma * 10) % 11;

  if ((resto == 10) || (resto == 11))
    resto = 0;

  if (resto != parseInt(cpf.substring(10, 11)))
    return false;

  return true;
}

function validarNome(nome) {

  const nomeLimpo = nome.trim();

  if (nomeLimpo.length < 3) {
    return false;
  }

  return true;
}

function Register() {

  const [name, setName] = useState("");
  const [cpf, setCpf] = useState("");
  const [dateBirth, setDateBirth] = useState("");
  const [successMessage, setSuccessMessage] = useState("");
  const [errorMessage, setErrorMessage] = useState("");

  const register = async (e) => {

    e.preventDefault();

    // limpar logs anteriores
    setTimeout(() => {
      setSuccessMessage("");
      setErrorMessage("");
    }, 4000);

    if (!validarNome(name)) {
    alert("Nome deve ter no mínimo 3 caracteres");
    return;
    }

    if (!validarCPF(cpf)) {
    alert("CPF deve conter 11 dígitos");
    return;
    }

    const formData = new FormData();
    formData.append("name", name);
    formData.append("cpf", cpf);
    formData.append("date_birth", dateBirth);

    try {

      await axios.post(
        "http://localhost:8000/register",
        formData
      );

    setSuccessMessage("Cadastro realizado com sucesso");
      setErrorMessage("");

    } catch (error) {

      console.log(error);

      setErrorMessage("Erro ao realizar cadastro, verifique os dados e tente novamente");
      setSuccessMessage("");

    }

  };

  // Função para formatar o CPF automaticamente
  const formatCpf = (value) => {
    // Remove tudo que não é número
    const cpfNumbers = value.replace(/\D/g, "");
    // Aplica a máscara
    return cpfNumbers
      .replace(/^(\d{3})(\d)/, "$1.$2")
      .replace(/^(\d{3})\.(\d{3})(\d)/, "$1.$2.$3")
      .replace(/^(\d{3})\.(\d{3})\.(\d{3})(\d)/, "$1.$2.$3-$4")
      .slice(0, 14); // Limita ao tamanho do CPF formatado
  };

  // Handler para o campo CPF
  const handleCpfChange = (e) => {
    const formatted = formatCpf(e.target.value);
    setCpf(formatted);
  };

  return (
    <div className="container">
      <div className="card">
        <h2>Cadastro da Paciente</h2>
        <form onSubmit={register}>
          <div className="form-group">
            <label>Nome</label>
            <input
              type="text"
              placeholder="Nome completo"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          </div>
          <div className="form-group">
            <label>CPF</label>
            <input
              type="text"
              placeholder="000.000.000-00"
              value={cpf}
              onChange={handleCpfChange}
              maxLength={14}
            />
          </div>
          <div className="form-group">
            <label>Data de nascimento</label>
            <input
              type="text"
              placeholder="DD/MM/AAAA"
              value={dateBirth}
              onChange={(e) => {
                // Formata a data automaticamente
                const value = e.target.value.replace(/\D/g, "");
                let formatted = value;
                if (value.length > 2) {
                  formatted = value.slice(0, 2) + "/" + value.slice(2);
                }
                if (value.length > 4) {
                  formatted = formatted.slice(0, 5) + "/" + value.slice(4, 8);
                }
                formatted = formatted.slice(0, 10);
                setDateBirth(formatted);
              }}
              maxLength={10}
            />
          </div>
          <button type="submit">
          Cadastrar
        </button>

        {successMessage && (
          <div className="success-message">
            {successMessage}
          </div>
        )}

        {errorMessage && (
          <div className="error-message">
            {errorMessage}
          </div>
        )}
        </form>
      </div>
    </div>
  );

}

export default Register;
