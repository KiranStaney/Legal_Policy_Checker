import React, { useState } from "react";
import axios from "axios";
import { marked } from "marked";

const BACKEND_URL = "http://localhost:8000";

export default function App() {
    const [file, setFile] = useState(null);
    const [uploadResponse, setUploadResponse] = useState(null);
    const [analysis, setAnalysis] = useState(null);
    const [chatInput, setChatInput] = useState("");
    const [chatResponse, setChatResponse] = useState("");
    const [loading, setLoading] = useState(false);

    // Select File
    const handleFileSelect = (event) => {
        setFile(event.target.files[0]);
    };

    // ----------------------------
    // STEP 1: UPLOAD DOCUMENT
    // ----------------------------
    const uploadDocument = async () => {
        if (!file) return alert("Please select a file first!");

        setLoading(true);

        const formData = new FormData();
        formData.append("file", file);

        try {
            const res = await axios.post(`${BACKEND_URL}/upload`, formData, {
                headers: { "Content-Type": "multipart/form-data" },
            });

            setUploadResponse(res.data);
        } catch (err) {
            console.error(err);
            alert("Upload failed: " + err.message);
        } finally {
            setLoading(false);
        }
    };

    // ---------------------------------------
    // STEP 2: FULL ANALYSIS
    // ---------------------------------------
    const runFullAnalysis = async () => {
        if (!uploadResponse) return alert("Upload a document first!");

        const payload = {
            document_id: uploadResponse.document_id,
            extracted_text: uploadResponse.extracted_text,
        };

        setLoading(true);

        try {
            const res = await axios.post(
                `${BACKEND_URL}/analysis/analyze-full`,
                payload,
                { headers: { "Content-Type": "application/json" } }
            );

            setAnalysis(res.data);
        } catch (err) {
            console.error(err);
            alert("Analysis Error: " + err.message);
        } finally {
            setLoading(false);
        }
    };


    const sendChat = async () => {
        if (!chatInput.trim()) return;

        const payload = {
            query: chatInput,
            context: uploadResponse?.extracted_text || "",
        };

        try {
            const res = await axios.post(`${BACKEND_URL}/chat`, payload, {
                headers: { "Content-Type": "application/json" },
            });

            setChatResponse(res.data.response);
        } catch (err) {
            console.error(err);
            alert("Chat Error: " + err.message);
        }
    };

    return (
        <div style={{ padding: "20px", fontFamily: "Arial" }}>
            <h1>Legal AI Assistant</h1>
            <div style={{ marginBottom: "20px" }}>
                <input type="file" onChange={handleFileSelect} />
                <button onClick={uploadDocument} disabled={loading}>
                    {loading ? "Uploading..." : "Upload Document"}
                </button>
            </div>
            {uploadResponse && (
                <div>
                    <h3>Upload Summary</h3>
                    <p><b>Document ID:</b> {uploadResponse.document_id}</p>

                    <div
                        className="summary-box"
                        dangerouslySetInnerHTML={{
                            __html: marked(uploadResponse.llm_summary || ""),
                        }}
                    />
                </div>
            )}

            <hr />

            <button onClick={runFullAnalysis} disabled={loading || !uploadResponse}>
                {loading ? "Analyzing..." : "Run Full Analysis"}
            </button>

            {analysis && (
                <div style={{ marginTop: "20px" }}>
                    <h2>Analysis Report</h2>
                    <p><b>Risk Score:</b> {analysis.final_risk_score}</p>

                    <h3>Issues Detected:</h3>
                    <ul>
                        {analysis.issues_detected.map((issue, idx) => (
                            <li key={idx}>
                                <b>{issue.clause_title}</b>: {issue.issue_description}
                            </li>
                        ))}
                    </ul>

                    <h3>Suggested Rewrites:</h3>
                    <ul>
                        {analysis.rewrite_suggestions.map((s, idx) => (
                            <li key={idx}>
                                <b>{s.original_text}</b> → {s.suggested_rewrite}
                            </li>
                        ))}
                    </ul>
                </div>
            )}

            <hr />

            <h2>Chat with AI</h2>
            <input
                type="text"
                placeholder="Ask something..."
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                style={{ width: "300px" }}
            />
            <button onClick={sendChat}>Send</button>

            {chatResponse && (
                <div style={{ marginTop: "15px" }}>
                    <h4>AI Response:</h4>
                    <p>{chatResponse}</p>
                </div>
            )}
        </div>
    );
}
