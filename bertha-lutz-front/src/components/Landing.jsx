import { useState, useRef } from "react";
import Register from "./Register";

function Landing() {
  const [step, setStep] = useState("video");
  const [started, setStarted] = useState(false);
  // "video" | "button" | "form"

  const videoRef = useRef(null);

  const handleVideoEnd = () => {
    setStep("button");
  };

return (
    <div className="overlay" style={{ display: "flex", flexDirection: "column", justifyContent: "flex-start", alignItems: "center", minHeight: "100vh", paddingTop: "20px" }}>

        {step === "video" && (
            <div className="container">
            {!started && (
            <button onClick={() => setStarted(true)} style={{ padding: "32px 64px", fontSize: "24px" }}>
                    Assista ao vídeo e cadastre-se!
            </button>
            )}
            </div>
        )}

        {!started && step === "video" && (
            <p style={{ marginTop: "12px", fontStyle: "italic", maxWidth: "600px", textAlign: "center" }}>
                "Nas mãos das mulheres está o coração da humanidade,<br/>
                a força que move montanhas,<br/>
                a doçura que cura feridas,<br/>
                a sabedoria que ilumina caminhos."<br/>
                <span style={{ marginTop: "12px", display: "block", fontStyle: "normal", fontSize: "14px" }}>
                    — Bertha Lutz
                </span>
            </p>
        )}

        {started && step === "video" && (
            <div className="container">
            <video
                    ref={videoRef}
                    width="1280"
                    height="720"
                    autoPlay
                    onEnded={handleVideoEnd}
                    controls
            >
                    <source src="/Video1.mp4" type="video/mp4" />
            </video>
            </div>
        )}

        {step === "button" && (
            <div className="container">
                <div className="card">
                    <h2>Bem-vinda</h2>
                    <button onClick={() => setStep("form")} style={{ padding: "16px 32px", fontSize: "18px" }}>
                        Cadastrar-se
                    </button>
                </div>
            </div>
        )}

        {step === "form" && <Register />}

    </div>
);
}

export default Landing;