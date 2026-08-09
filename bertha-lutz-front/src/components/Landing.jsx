import { useState } from "react";
import Register from "./Register";

const STEPS = {
  intro: "intro",
  video: "video",
  welcome: "welcome",
  form: "form",
};

const QUOTATION = (
  <p className="quotation">
    "Nas mãos das mulheres está o coração da humanidade,
    <br />
    a força que move montanhas, a doçura que cura feridas,
    <br />
    a sabedoria que ilumina caminhos."
    <span>— Bertha Lutz</span>
  </p>
);

function Landing() {
  const [step, setStep] = useState(STEPS.intro);

  return (
    <div className="overlay">
      <div className="landing">
        {step === STEPS.intro && (
          <>
            {QUOTATION}
            <button
              className="btn btn-start"
              onClick={() => setStep(STEPS.video)}
            >
              Assista ao vídeo e cadastre-se!
            </button>
          </>
        )}

        {step === STEPS.video && (
          <video
            className="landing-video"
            width="1280"
            height="720"
            autoPlay
            controls
            onEnded={() => setStep(STEPS.welcome)}
          >
            <source src="/Video1.mp4" type="video/mp4" />
          </video>
        )}

        {step === STEPS.welcome && (
          <div className="card">
            <h2>Bem-vinda</h2>
            <button className="btn" onClick={() => setStep(STEPS.form)}>
              Cadastrar-se
            </button>
          </div>
        )}

        {step === STEPS.form && <Register />}
      </div>
    </div>
  );
}

export default Landing;