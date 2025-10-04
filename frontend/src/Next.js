import React, { useState, useEffect } from "react";
import Select from "react-select";
import "./Next.css";

export default function SymptomSelector() {
  const [symptomOptions, setSymptomOptions] = useState([]);
  const [selectedSymptoms, setSelectedSymptoms] = useState([]);
  const [textSymptoms, setTextSymptoms] = useState("");
  const [submitted, setSubmitted] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  // Fetch symptoms from Flask backend
  useEffect(() => {
    fetch("http://127.0.0.1:5000/symptoms")
      .then((res) => res.json())
      .then((data) =>
        setSymptomOptions(
          data.map((sym) => ({ label: sym.display, value: sym.value }))
        )
      )
      .catch((err) => console.error("Error fetching symptoms:", err));
  }, []);

  // Handle submit
  const handleSubmit = async () => {
    setError("");
    if (selectedSymptoms.length === 0 && textSymptoms.trim() === "") {
      setError("Please select or enter at least one symptom.");
      return;
    }

    const payload = {
      symptoms: [...selectedSymptoms.map((s) => s.value), textSymptoms].join(
        ", "
      ),
    };

    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json();

      if (data.error) {
        setError(data.error);
        setResult(null);
      } else {
        setResult(
          data.recommended_specialty
            ? `${data.predicted_disease} (Specialist: ${data.recommended_specialty})`
            : data.predicted_disease
        );
        setSubmitted(true);
      }
    } catch (err) {
      console.error("Error submitting symptoms:", err);
      setError("Error connecting to backend.");
      setResult(null);
    }
  };

  const handleClear = () => {
    setSelectedSymptoms([]);
    setTextSymptoms("");
    setResult(null);
    setSubmitted(false);
    setError("");
  };

  return (
    <div className="next-page">
      <div className="glass-box">
        <h2>Select or Enter Your Symptoms</h2>

        <div className="input-row">
          <Select
            isMulti
            options={symptomOptions}
            value={selectedSymptoms}
            onChange={setSelectedSymptoms}
            placeholder="Select symptoms..."
            className="dropdown"
            styles={{
              option: (provided, state) => ({
                ...provided,
                color: "black",
                backgroundColor: state.isFocused ? "#f0f0f0" : "white",
              }),
              singleValue: (provided) => ({
                ...provided,
                color: "black",
              }),
              multiValueLabel: (provided) => ({
                ...provided,
                color: "black",
              }),
            }}
          />

          <textarea
            placeholder="Or type your symptoms here..."
            value={textSymptoms}
            onChange={(e) => setTextSymptoms(e.target.value)}
            className="text-box"
          />

          {!submitted ? (
            <button onClick={handleSubmit} className="submit-btn">
              Submit
            </button>
          ) : (
            <button onClick={handleClear} className="clear-btn">
              Clear
            </button>
          )}
        </div>

        {error && <p className="error-text">{error}</p>}

        {submitted && result && (
          <div className="results-box">
            {selectedSymptoms.length > 0 && (
              <>
                <h3>Selected Symptoms:</h3>
                <p>{selectedSymptoms.map((s) => s.label).join(", ")}</p>
              </>
            )}

            {textSymptoms.trim() !== "" && (
              <>
                <h3>Text Symptoms:</h3>
                <p>{textSymptoms}</p>
              </>
            )}

            <h3>Prediction:</h3>
            <p>{result}</p>
          </div>
        )}
      </div>
    </div>
  );
}
