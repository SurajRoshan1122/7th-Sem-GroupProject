import React from "react";
import { useNavigate } from "react-router-dom";
import "./Home.css";

function Home() {
  const navigate = useNavigate();

  const handleClick = () => {
    navigate("/next"); // takes user to Next page
  };

  return (
    <div className="home">
      <div className="section">
        <h1>AI Symptom Detector</h1>
        <button onClick={handleClick}>Get Started!</button>
      </div>
    </div>
  );
}

export default Home;
