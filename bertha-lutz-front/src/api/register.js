import axios from "axios";
import { API_BASE_URL } from "../config";

export async function registerUser({ name, cpf, dateBirth, phone }) {
  const formData = new FormData();
  formData.append("name", name);
  formData.append("cpf", cpf);
  formData.append("date_birth", dateBirth);
  formData.append("phone", phone);

  const { data } = await axios.post(`${API_BASE_URL}/register`, formData);
  return data;
}