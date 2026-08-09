import { useState } from "react";
import { registerUser } from "../api/register";
import { validarFormulario } from "../utils/validation";
import { formatCpf, formatDate, formatPhone } from "../utils/masks";

function Register() {
  const [form, setForm] = useState({ name: "", cpf: "", dateBirth: "", phone: "" });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [registered, setRegistered] = useState(false);
  const [error, setError] = useState("");

  const handleChange = (field, mask) => (event) => {
    const value = mask ? mask(event.target.value) : event.target.value;
    setForm((prev) => ({ ...prev, [field]: value }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError("");

    const validationError = validarFormulario(form);
    if (validationError) {
      setError(validationError);
      return;
    }

    setIsSubmitting(true);
    try {
      await registerUser(form);
      setRegistered(true);
    } catch (err) {
      setError(err.response?.data?.detail || "Erro ao realizar cadastro, verifique os dados e tente novamente");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="container">
      <div className="card">
        <h2>Cadastro da Paciente</h2>
        <form onSubmit={handleSubmit} noValidate>
          <div className="form-group">
            <label htmlFor="name">Nome</label>
            <input
              id="name"
              type="text"
              placeholder="Nome completo"
              value={form.name}
              onChange={handleChange("name")}
              disabled={registered}
            />
          </div>
          <div className="form-group">
            <label htmlFor="cpf">CPF</label>
            <input
              id="cpf"
              type="text"
              placeholder="000.000.000-00"
              value={form.cpf}
              onChange={handleChange("cpf", formatCpf)}
              maxLength={14}
              disabled={registered}
            />
          </div>
          <div className="form-group">
            <label htmlFor="birthDate">Data de nascimento</label>
            <input
              id="birthDate"
              type="text"
              placeholder="DD/MM/AAAA"
              value={form.dateBirth}
              onChange={handleChange("dateBirth", formatDate)}
              maxLength={10}
              disabled={registered}
            />
          </div>
          <div className="form-group">
            <label htmlFor="phone">Telefone</label>
            <input
              id="phone"
              type="tel"
              placeholder="(00) 00000-0000"
              value={form.phone}
              onChange={handleChange("phone", formatPhone)}
              maxLength={15}
              disabled={registered}
            />
          </div>

          <button
            type="submit"
            className={registered ? "btn btn-registered" : "btn"}
            disabled={isSubmitting || registered}
          >
            {registered ? "Cadastrado" : isSubmitting ? "Cadastrando..." : "Cadastrar"}
          </button>

          {error && (
            <div className="error-message" role="alert">
              {error}
            </div>
          )}
          {registered && (
            <div className="success-message">Cadastro realizado com sucesso!</div>
          )}
        </form>
      </div>
    </div>
  );
}

export default Register;