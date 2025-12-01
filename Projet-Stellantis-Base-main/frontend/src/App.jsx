import React, { useState, useEffect, useCallback, useRef } from 'react';
// Correction des chemins d'importation :
import CameraView from './components/CameraView';
import { useCamera } from './hooks/useCamera';
import './App.css';

const PREDICT_URL = 'http://localhost:5000/predict';

function App() {
  const { videoRef, canvasRef, captureFrame, error } = useCamera();

  const [temperature, setTemperature] = useState(20);
  const temperatureRef = useRef(temperature);
  const [currentEmotion, setCurrentEmotion] = useState('');
  const [annotatedImage, setAnnotatedImage] = useState(null);
  const [primaryEmotion, setPrimaryEmotion] = useState('');
  const [vlmQuestion, setVlmQuestion] = useState(null);
  const [isWaitingVLM, setIsWaitingVLM] = useState(false);
  const [lastVLMCheck, setLastVLMCheck] = useState(Date.now());

  useEffect(() => {
    temperatureRef.current = temperature;
  }, [temperature]);

  const processFrame = useCallback(() => {
    const video = videoRef.current;
    if (!video || video.readyState !== 4) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(async (blob) => {
        if (!blob) {
            console.error("Échec de la conversion en Blob.");
            return;
        }

        const formData = new FormData();
        formData.append('image', blob, 'capture.png');

        try {
            const response = await fetch(PREDICT_URL, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`Erreur HTTP: ${response.status}`);
            }

            const result = await response.json();

            setPrimaryEmotion(result.emotion);

            const action = result.comfort_action;

            if (action === "AUGMENTER_TEMP" && temperatureRef.current < 25) {
                setTemperature(prev => Math.min(prev + 1, 25));
                console.log(`Action: ${action}. T° ajustée.`);
            } else if (action === "BAISSER_VENTILATION") {
                console.log(`Action: ${action}. Ventilation à ajuster.`);
            }

        } catch (err) {
            console.error('Erreur communication avec l\'API Python:', err);
        }
    }, 'image/jpeg', 0.8);
  }, [videoRef, canvasRef]);

  // VLM Check (désactivé)
  const checkVLM = useCallback(async () => {
    if (isWaitingVLM) return;

    const elapsed = Date.now() - lastVLMCheck;
    if (elapsed < 5000) return;

    setLastVLMCheck(Date.now());

  }, [isWaitingVLM, lastVLMCheck]);

  const handleVLMResponse = async (response) => {
    console.log('[VLM Response] User clicked:', response);

    setVlmQuestion(null);
    setIsWaitingVLM(false);
    setLastVLMCheck(Date.now());
  };

  // CNN Analysis à 2 FPS pour la stabilité
  useEffect(() => {
    const frameInterval = setInterval(processFrame, 500);
    return () => clearInterval(frameInterval);
  }, [processFrame]);

  // VLM Check toutes les secondes (désactivé)
  useEffect(() => {
    return () => {};
  }, [checkVLM]);

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-content">
          <div className="logo-space">
            <img src="/Stellantis.png" alt="Stellantis" />
          </div>
          <h1 className="app-title">CARE</h1>
          <div></div>
        </div>
      </header>

      <div className="app-content">
        <CameraView
          videoRef={videoRef}
          canvasRef={canvasRef}
          annotatedImage={annotatedImage}
          emotion={currentEmotion}
          error={error}
          vlmQuestion={vlmQuestion}
          onVLMResponse={handleVLMResponse}
          temperature={temperature}
          primaryEmotion={primaryEmotion}
        />
      </div>
    </div>
  );
}

export default App;