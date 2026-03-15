import { useState } from "react";
import axios from "axios";

function Register() {

  const [name, setName] = useState("");
  const [cpf, setCpf] = useState("");

  const register = async (e) => {

    e.preventDefault();

    const formData = new FormData();
    formData.append("name", name);
    formData.append("cpf", cpf);

    try {

      await axios.post(
        "http://localhost:8000/register",
        formData
      );

      alert("Cadastro realizado");

    } catch (error) {

      console.log(error);
      alert("Erro no cadastro");

    }

  };

  return (

    <div>

      <h2>Cadastro</h2>

      <form onSubmit={register}>

        <div>
          Nome
          <input
            type="text"
            value={name}
            onChange={(e)=>setName(e.target.value)}
          />
        </div>

        <div>
          CPF
          <input
            type="text"
            value={cpf}
            onChange={(e)=>setCpf(e.target.value)}
          />
        </div>

        <button type="submit">
          Cadastrar
        </button>

      </form>

    </div>

  );

}

export default Register;
