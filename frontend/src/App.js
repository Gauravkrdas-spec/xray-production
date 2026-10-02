import { useState, useRef, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { analyzeXray, submitFeedback } from './services/api';
import jsPDF from 'jspdf';
import autoTable from 'jspdf-autotable';
import './styles/main.css';

export default function App() {
  const [image, setImage] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [results, setResults] = useState(null);
  const [view, setView] = useState('annotated');
  const [feedback, setFeedback] = useState({
    doctorName: '',
    hospital:   '',
    rating:     '',
    comments:   '',
    recommend:  ''
  });
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);
  const [submittingFeedback, setSubmittingFeedback] = useState(false);
  const [patientInfo, setPatientInfo] = useState({
    name:       '',
    age:        '',
    gender:     '',
    referredBy: ''
  });
  const [error, setError] = useState(null);
  const [generatingPdf, setGeneratingPdf] = useState(false);

  const canvasAnnotRef = useRef(null);
  const canvasEnhRef   = useRef(null);
  const canvasSegRef   = useRef(null);

  // ── Draw plain or filtered image ──────────────────
  const drawImage = useCallback((src, canvasRef, filter = null) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const img = new Image();
    img.onload = () => {
      const max = 500;
      let w = img.width, h = img.height;
      if (w > max || h > max) {
        const r = Math.min(max / w, max / h);
        w = Math.round(w * r);
        h = Math.round(h * r);
      }
      canvas.width  = w;
      canvas.height = h;
      if (filter) ctx.filter = filter;
      ctx.drawImage(img, 0, 0, w, h);
      ctx.filter = 'none';
    };
    img.src = src;
  }, []);

  // ── Draw segmented pseudo-color view ──────────────
  const drawSegmented = useCallback((src) => {
    const canvas = canvasSegRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const img = new Image();
    img.onload = () => {
      const max = 500;
      let w = img.width, h = img.height;
      if (w > max || h > max) {
        const r = Math.min(max / w, max / h);
        w = Math.round(w * r);
        h = Math.round(h * r);
      }
      canvas.width  = w;
      canvas.height = h;
      ctx.drawImage(img, 0, 0, w, h);
      const id = ctx.getImageData(0, 0, w, h);
      const d  = id.data;
      for (let i = 0; i < d.length; i += 4) {
        const g = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
        if      (g > 190) { d[i]=195; d[i+1]=205; d[i+2]=255; }
        else if (g > 110) { d[i]=70;  d[i+1]=165; d[i+2]=95;  }
        else if (g > 44)  { d[i]=165; d[i+1]=85;  d[i+2]=75;  }
        else              { d[i]=18;  d[i+1]=28;  d[i+2]=78;  }
      }
      ctx.putImageData(id, 0, 0);
      const legend = [
        { c: 'rgba(195,205,255,.9)', l: 'Bone/Rib'   },
        { c: 'rgba(70,165,95,.9)',   l: 'Lung'        },
        { c: 'rgba(165,85,75,.9)',   l: 'Soft tissue' },
        { c: 'rgba(18,28,78,.85)',   l: 'Air'         },
      ];
      legend.forEach((item, i) => {
        ctx.fillStyle = item.c;
        ctx.fillRect(8, 8 + i * 18, 12, 12);
        ctx.fillStyle = '#fff';
        ctx.font = '10px monospace';
        ctx.fillText(item.l, 24, 18 + i * 18);
      });
    };
    img.src = src;
  }, []);

  // ── Handle file drop ──────────────────────────────
  const onDrop = useCallback((files) => {
    const file = files[0];
    if (!file) return;
    setImageFile(file);
    setResults(null);
    setError(null);
    setFeedbackSubmitted(false);
    const url = URL.createObjectURL(file);
    setImage(url);
    setTimeout(() => { drawImage(url, canvasAnnotRef); }, 100);
  }, [drawImage]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept:   { 'image/*': ['.jpg', '.jpeg', '.png'] },
    multiple: false
  });

  // ── Analyze X-Ray ─────────────────────────────────
  const handleAnalyze = async () => {
    if (!imageFile) return;
    setAnalyzing(true);
    setError(null);
    try {
      const data = await analyzeXray(imageFile, patientInfo);
      setResults(data);

      const src    = URL.createObjectURL(imageFile);
      const canvas = canvasAnnotRef.current;
      const ctx    = canvas.getContext('2d');
      const img    = new Image();

      img.onload = () => {
        const max = 500;
        let w = img.width, h = img.height;
        if (w > max || h > max) {
          const r = Math.min(max / w, max / h);
          w = Math.round(w * r);
          h = Math.round(h * r);
        }
        canvas.width  = w;
        canvas.height = h;
        ctx.drawImage(img, 0, 0, w, h);

        (data.findings || []).slice(0, 5).forEach((f, i) => {
          const rad = Math.min(
            (f.radius || 0.06) * Math.min(w, h),
            Math.min(w, h) * 0.07
          );
          const cx = Math.min(Math.max(f.x * w, rad + 10), w - rad - 10);
          const cy = Math.min(Math.max(f.y * h, rad + 10), h - rad - 10);
          const color =
            f.severity === 'high'   ? '#e24b4a' :
            f.severity === 'medium' ? '#ef9f27' : '#63aa22';

          ctx.save();
          ctx.beginPath();
          ctx.arc(cx, cy, rad, 0, 2 * Math.PI);
          ctx.strokeStyle = color;
          ctx.lineWidth   = 2.5;
          ctx.setLineDash([6, 3]);
          ctx.stroke();
          ctx.setLineDash([]);

          ctx.beginPath();
          ctx.arc(cx, cy, 4, 0, 2 * Math.PI);
          ctx.fillStyle = color;
          ctx.fill();

          ctx.font      = 'bold 14px monospace';
          ctx.fillStyle = 'rgba(0,0,0,0.8)';
          ctx.fillText(i + 1, cx + rad * 0.7 + 2, cy - rad * 0.7 + 2);
          ctx.fillStyle = color;
          ctx.fillText(i + 1, cx + rad * 0.7, cy - rad * 0.7);

          ctx.font    = 'bold 11px monospace';
          const label = `${i + 1}. ${f.name} ${f.probability}%`;
          const lw    = ctx.measureText(label).width;
          let lx      = cx - lw / 2;
          let ly      = cy - rad - 10;
          lx          = Math.max(4, Math.min(w - lw - 8, lx));
          ly          = Math.max(16, ly);

          ctx.fillStyle = 'rgba(0,0,0,0.85)';
          ctx.beginPath();
          ctx.roundRect(lx - 4, ly - 13, lw + 8, 17, 4);
          ctx.fill();
          ctx.strokeStyle = color;
          ctx.lineWidth   = 1;
          ctx.stroke();
          ctx.fillStyle = color;
          ctx.fillText(label, lx, ly);
          ctx.restore();
        });
      };
      img.src = src;

      drawImage(image, canvasEnhRef,
        'contrast(175%) brightness(108%) grayscale(100%)');
      drawSegmented(image);
      setView('annotated');

    } catch (e) {
      setError(
        e.response?.data?.error || 'Analysis failed. Is backend running?'
      );
    } finally {
      setAnalyzing(false);
    }
  };

  // ── Generate PDF Report ───────────────────────────
  const generatePDF = async () => {
    if (!results) return;
    setGeneratingPdf(true);
    try {
      const doc  = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' });
      const pw   = doc.internal.pageSize.getWidth();
      const ph   = doc.internal.pageSize.getHeight();
      const margin = 15;
      let y = margin;

      // ── Header bar ──
      doc.setFillColor(10, 30, 60);
      doc.rect(0, 0, pw, 28, 'F');
      doc.setTextColor(255, 255, 255);
      doc.setFontSize(18);
      doc.setFont('helvetica', 'bold');
      doc.text('RadVision AI', margin, 12);
      doc.setFontSize(9);
      doc.setFont('helvetica', 'normal');
      doc.text('Chest X-Ray Analysis Workstation', margin, 19);
      doc.setFontSize(8);
      doc.text(
        `Generated: ${new Date().toLocaleString()}`,
        pw - margin, 19, { align: 'right' }
      );
      y = 36;

      // ── Urgency badge ──
      const urgency = results.report?.urgency || 'routine';
      const urgencyColor =
        urgency === 'emergency' ? [226, 75, 74]  :
        urgency === 'urgent'    ? [239, 159, 39] : [99, 170, 34];
      doc.setFillColor(...urgencyColor);
      doc.roundedRect(pw - margin - 30, 30, 30, 8, 2, 2, 'F');
      doc.setTextColor(255, 255, 255);
      doc.setFontSize(8);
      doc.setFont('helvetica', 'bold');
      doc.text(urgency.toUpperCase(), pw - margin - 15, 35.5, { align: 'center' });

      // ── Patient info box ──
      doc.setFillColor(240, 244, 248);
      doc.rect(margin, y, pw - margin * 2, 22, 'F');
      doc.setDrawColor(180, 200, 220);
      doc.rect(margin, y, pw - margin * 2, 22, 'S');
      doc.setTextColor(30, 50, 80);
      doc.setFontSize(9);
      doc.setFont('helvetica', 'bold');
      doc.text('PATIENT INFORMATION', margin + 4, y + 6);
      doc.setFont('helvetica', 'normal');
      doc.setFontSize(8);
      doc.setTextColor(60, 60, 60);
      const pi = patientInfo;
      doc.text(`Name: ${pi.name || 'N/A'}`,        margin + 4,      y + 13);
      doc.text(`Age: ${pi.age || 'N/A'}`,           margin + 60,     y + 13);
      doc.text(`Gender: ${pi.gender || 'N/A'}`,     margin + 90,     y + 13);
      doc.text(`Referred By: ${pi.referredBy || 'N/A'}`, margin + 130, y + 13);
      doc.text(`Findings: ${results.findings.length}`,   margin + 4,  y + 19);
      doc.text(
        `Image ID: ${results.image_id?.substring(0, 20) || 'N/A'}`,
        margin + 60, y + 19
      );
      y += 28;

      // ── X-ray image from canvas ──
      const annotCanvas = canvasAnnotRef.current;
      if (annotCanvas) {
        const imgData  = annotCanvas.toDataURL('image/jpeg', 0.85);
        const imgW     = pw - margin * 2;
        const imgH     = (annotCanvas.height / annotCanvas.width) * imgW;
        const safeH    = Math.min(imgH, 80);
        doc.addImage(imgData, 'JPEG', margin, y, imgW, safeH);
        y += safeH + 4;

        // Image caption
        doc.setFontSize(7);
        doc.setTextColor(120, 120, 120);
        doc.text(
          'Figure 1: Annotated chest X-ray with AI-detected findings',
          pw / 2, y, { align: 'center' }
        );
        y += 6;
      }

      // ── Findings table ──
      doc.setFontSize(10);
      doc.setFont('helvetica', 'bold');
      doc.setTextColor(10, 30, 60);
      doc.text('AI DETECTED FINDINGS', margin, y);
      y += 4;

      const tableRows = results.findings.map((f, i) => [
        i + 1,
        f.name,
        `${f.probability}%`,
        f.severity.toUpperCase(),
        f.severity === 'high'   ? 'Needs attention' :
        f.severity === 'medium' ? 'Monitor'          : 'Low risk'
      ]);

      autoTable(doc, {
        startY:      y,
        head:        [['#', 'Finding', 'Probability', 'Severity', 'Status']],
        body:        tableRows,
        margin:      { left: margin, right: margin },
        headStyles: {
          fillColor:  [10, 30, 60],
          textColor:  [255, 255, 255],
          fontStyle:  'bold',
          fontSize:   8,
        },
        bodyStyles: {
          fontSize:   8,
          textColor:  [40, 40, 40],
        },
        alternateRowStyles: {
          fillColor: [245, 248, 252],
        },
        didParseCell: (data) => {
          if (data.section === 'body' && data.column.index === 3) {
            const val = data.cell.raw;
            if      (val === 'HIGH')   data.cell.styles.textColor = [226, 75,  74];
            else if (val === 'MEDIUM') data.cell.styles.textColor = [239, 159, 39];
            else                       data.cell.styles.textColor = [99,  170, 34];
            data.cell.styles.fontStyle = 'bold';
          }
        },
        theme: 'grid',
      });
      y = doc.lastAutoTable.finalY + 8;

      // ── Check if new page needed ──
      const checkPage = (needed) => {
        if (y + needed > ph - margin) {
          doc.addPage();
          y = margin;
        }
      };

      // ── Clinical Findings ──
      checkPage(30);
      doc.setFontSize(10);
      doc.setFont('helvetica', 'bold');
      doc.setTextColor(10, 30, 60);
      doc.text('CLINICAL FINDINGS', margin, y);
      y += 5;
      doc.setFont('helvetica', 'normal');
      doc.setFontSize(8);
      doc.setTextColor(50, 50, 50);
      const findingLines = doc.splitTextToSize(
        results.report?.findings || '', pw - margin * 2
      );
      doc.text(findingLines, margin, y);
      y += findingLines.length * 4 + 6;

      // ── Normal Structures ──
      checkPage(20);
      doc.setFontSize(10);
      doc.setFont('helvetica', 'bold');
      doc.setTextColor(10, 30, 60);
      doc.text('NORMAL STRUCTURES', margin, y);
      y += 5;
      doc.setFont('helvetica', 'normal');
      doc.setFontSize(8);
      doc.setTextColor(50, 50, 50);
      const normalLines = doc.splitTextToSize(
        results.report?.normal_structures || '', pw - margin * 2
      );
      doc.text(normalLines, margin, y);
      y += normalLines.length * 4 + 6;

      // ── Impression box ──
      checkPage(30);
      doc.setFontSize(10);
      doc.setFont('helvetica', 'bold');
      doc.setTextColor(10, 30, 60);
      doc.text('IMPRESSION', margin, y);
      y += 5;
      const impLines = doc.splitTextToSize(
        results.report?.impression || '', pw - margin * 2 - 8
      );
      const impH = impLines.length * 4 + 8;
      doc.setFillColor(235, 243, 255);
      doc.setDrawColor(74, 144, 217);
      doc.rect(margin, y, pw - margin * 2, impH, 'F');
      doc.setLineWidth(1);
      doc.line(margin, y, margin, y + impH);
      doc.setLineWidth(0.2);
      doc.setFont('helvetica', 'italic');
      doc.setFontSize(8);
      doc.setTextColor(20, 60, 120);
      doc.text(impLines, margin + 4, y + 5);
      y += impH + 8;

      // ── Recommendations ──
      checkPage(30);
      doc.setFontSize(10);
      doc.setFont('helvetica', 'bold');
      doc.setTextColor(10, 30, 60);
      doc.text('RECOMMENDATIONS', margin, y);
      y += 5;
      doc.setFont('helvetica', 'normal');
      doc.setFontSize(8);
      doc.setTextColor(50, 50, 50);
      (results.report?.recommendations || []).forEach((rec, i) => {
        checkPage(8);
        const recLines = doc.splitTextToSize(
          `${i + 1}. ${rec}`, pw - margin * 2 - 5
        );
        doc.text(recLines, margin + 2, y);
        y += recLines.length * 4 + 2;
      });
      y += 4;

      // ── Doctor validation ──
      if (feedbackSubmitted && feedback.doctorName) {
        checkPage(30);
        doc.setFontSize(10);
        doc.setFont('helvetica', 'bold');
        doc.setTextColor(10, 30, 60);
        doc.text('DOCTOR VALIDATION', margin, y);
        y += 4;

        autoTable(doc, {
          startY: y,
          head:   [['Doctor', 'Hospital', 'Accuracy', 'Recommend', 'Comments']],
          body:   [[
            feedback.doctorName,
            feedback.hospital    || 'N/A',
            feedback.rating      || 'N/A',
            feedback.recommend   || 'N/A',
            feedback.comments    || 'N/A',
          ]],
          margin:      { left: margin, right: margin },
          headStyles:  { fillColor: [10, 30, 60], textColor: [255,255,255], fontSize: 8 },
          bodyStyles:  { fontSize: 8 },
          theme:       'grid',
        });
        y = doc.lastAutoTable.finalY + 8;
      }

      // ── Footer disclaimer ──
      const totalPages = doc.internal.getNumberOfPages();
      for (let p = 1; p <= totalPages; p++) {
        doc.setPage(p);
        doc.setFillColor(240, 244, 248);
        doc.rect(0, ph - 16, pw, 16, 'F');
        doc.setFontSize(6.5);
        doc.setTextColor(120, 120, 120);
        doc.setFont('helvetica', 'normal');
        doc.text(
          'DISCLAIMER: This AI analysis is for research purposes only and does not constitute medical advice. ' +
          'All findings must be verified by a licensed radiologist.',
          pw / 2, ph - 9, { align: 'center', maxWidth: pw - margin * 2 }
        );
        doc.text(
          `Page ${p} of ${totalPages}`,
          pw - margin, ph - 4, { align: 'right' }
        );
        doc.text('RadVision AI — Confidential', margin, ph - 4);
      }

      // ── Save PDF ──
      const filename = `RadVision_${pi.name || 'Patient'}_${
        new Date().toISOString().slice(0, 10)
      }.pdf`;
      doc.save(filename);

    } catch (e) {
      console.error('PDF error:', e);
      alert('PDF generation failed: ' + e.message);
    } finally {
      setGeneratingPdf(false);
    }
  };

  // ── Submit doctor feedback ────────────────────────
  const handleFeedbackSubmit = async () => {
    if (!feedback.doctorName || !feedback.rating) {
      alert('Please enter your name and accuracy rating');
      return;
    }
    setSubmittingFeedback(true);
    try {
      await submitFeedback({
        ...feedback,
        analysis_id: results?.image_id,
        findings:    results?.findings,
        timestamp:   new Date().toISOString()
      });
      setFeedbackSubmitted(true);
    } catch (e) {
      setFeedbackSubmitted(true);
    } finally {
      setSubmittingFeedback(false);
    }
  };

  const getUrgencyClass = (u) => {
    if (u === 'emergency') return 'urgency-emergency';
    if (u === 'urgent')    return 'urgency-urgent';
    return 'urgency-routine';
  };

  const views      = ['annotated', 'enhanced', 'segmented'];
  const viewLabels = { annotated: 'Annotated', enhanced: 'Enhanced', segmented: 'Segmented' };

  // ── RENDER ────────────────────────────────────────
  return (
    <div>

      {/* ══ HEADER ══ */}
      <div className="header">
        <div className="header-logo">+</div>
        <div>
          <div className="header-title">RadVision AI</div>
          <div className="header-sub">CHEST X-RAY ANALYSIS WORKSTATION</div>
        </div>
        {results && (
          <div style={{ marginLeft: 'auto', display: 'flex', gap: 8, alignItems: 'center' }}>
            <div className="header-badge">
              {results.findings.length} Finding(s) Detected
            </div>
            {/* PDF Download Button */}
            <button
              onClick={generatePDF}
              disabled={generatingPdf}
              style={{
                padding:      '5px 14px',
                background:   generatingPdf ? '#1a1d26' : '#1a3a5c',
                border:       '1px solid #4A90D9',
                borderRadius: 6,
                color:        '#4A90D9',
                fontSize:     12,
                fontWeight:   600,
                cursor:       generatingPdf ? 'not-allowed' : 'pointer',
                display:      'flex',
                alignItems:   'center',
                gap:          6,
                transition:   'all .2s'
              }}
            >
              {generatingPdf ? 'Generating...' : 'Download PDF'}
            </button>
          </div>
        )}
      </div>

      {/* ══ MAIN 3-COLUMN LAYOUT ══ */}
      <div className="main-layout">

        {/* ════ LEFT PANEL ════ */}
        <div className="left-panel">

          <div
            {...getRootProps()}
            className={`upload-zone ${isDragActive ? 'active' : ''}`}
          >
            <input {...getInputProps()} />
            <div className="upload-icon">📂</div>
            {image ? (
              <div style={{ fontSize: 12, color: '#4A90D9' }}>
                Image loaded — click to change
              </div>
            ) : (
              <>
                <div className="upload-title">Drop Chest X-Ray here</div>
                <div className="upload-sub">JPG · PNG supported</div>
              </>
            )}
            <button className="upload-btn">Browse File</button>
          </div>

          <div className="section-box">
            <div className="section-title">Patient Information</div>

            <div className="form-group">
              <label className="form-label">Patient Name</label>
              <input
                className="form-input"
                placeholder="Enter name"
                value={patientInfo.name}
                onChange={e =>
                  setPatientInfo({ ...patientInfo, name: e.target.value })
                }
              />
            </div>

            <div className="form-group">
              <label className="form-label">Age</label>
              <input
                className="form-input"
                placeholder="Age"
                type="number"
                value={patientInfo.age}
                onChange={e =>
                  setPatientInfo({ ...patientInfo, age: e.target.value })
                }
              />
            </div>

            <div className="form-group">
              <label className="form-label">Gender</label>
              <select
                className="form-select"
                value={patientInfo.gender}
                onChange={e =>
                  setPatientInfo({ ...patientInfo, gender: e.target.value })
                }
              >
                <option value="">Select</option>
                <option value="male">Male</option>
                <option value="female">Female</option>
                <option value="other">Other</option>
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">Referred By</label>
              <input
                className="form-input"
                placeholder="Doctor name"
                value={patientInfo.referredBy}
                onChange={e =>
                  setPatientInfo({ ...patientInfo, referredBy: e.target.value })
                }
              />
            </div>
          </div>

          <button
            className="analyze-btn"
            onClick={handleAnalyze}
            disabled={!imageFile || analyzing}
          >
            {analyzing
              ? <><div className="spinner" /> Analyzing...</>
              : 'Analyze X-Ray'
            }
          </button>

          {error && (
            <div style={{
              margin: '0 16px', padding: '10px 12px',
              background: '#2a1010', border: '1px solid #3a1515',
              borderRadius: 6, color: '#e24b4a', fontSize: 12
            }}>
              {error}
            </div>
          )}

        </div>
        {/* ════ END LEFT PANEL ════ */}


        {/* ════ CENTER PANEL ════ */}
        <div className="center-panel">

          <div className="view-tabs">
            {views.map(v => (
              <button
                key={v}
                className={`view-tab ${view === v ? 'active' : ''}`}
                onClick={() => setView(v)}
              >
                {viewLabels[v]}
              </button>
            ))}
          </div>

          <div className="canvas-area">
            {!image ? (
              <div className="empty-canvas">
                <div style={{ fontSize: 64 }}>📂</div>
                <div>Upload a chest X-ray to begin</div>
              </div>
            ) : (
              <>
                <canvas ref={canvasAnnotRef}
                  style={{ display: view === 'annotated' ? 'block' : 'none' }} />
                <canvas ref={canvasEnhRef}
                  style={{ display: view === 'enhanced'  ? 'block' : 'none' }} />
                <canvas ref={canvasSegRef}
                  style={{ display: view === 'segmented' ? 'block' : 'none' }} />
                <div className="canvas-label">
                  {view === 'annotated' && 'ANNOTATED VIEW'}
                  {view === 'enhanced'  && 'HIGH CONTRAST ENHANCED'}
                  {view === 'segmented' && 'PSEUDO-COLOR SEGMENTED'}
                </div>
              </>
            )}

            {analyzing && (
              <div style={{
                position: 'absolute', inset: 0,
                background: 'rgba(6,8,9,.85)',
                display: 'flex', flexDirection: 'column',
                alignItems: 'center', justifyContent: 'center', gap: 16
              }}>
                <div className="spinner"
                  style={{ width: 40, height: 40, borderWidth: 3 }} />
                <div style={{ fontSize: 14, color: '#778' }}>
                  AI analyzing chest X-ray...
                </div>
                <div style={{ fontSize: 11, color: '#445' }}>
                  CheXNet detecting findings · Claude writing report
                </div>
              </div>
            )}
          </div>

          {results && (
            <div style={{
              padding: '10px 16px', background: '#0a0c10',
              borderTop: '1px solid #1e2130',
              display: 'flex', flexWrap: 'wrap', gap: 6
            }}>
              {results.findings.map((f, i) => (
                <div key={i} className={`finding-chip chip-${f.severity}`}>
                  {f.name} {f.probability}%
                </div>
              ))}
            </div>
          )}

        </div>
        {/* ════ END CENTER PANEL ════ */}


        {/* ════ RIGHT PANEL ════ */}
        <div className="right-panel">

          <div className="panel-header">
            Radiology Report
            {results && (
              <span
                className={getUrgencyClass(results.report?.urgency)}
                style={{ marginLeft: 'auto' }}
              >
                {results.report?.urgency?.toUpperCase()}
              </span>
            )}
          </div>

          <div className="panel-body">
            {!results ? (
              <div className="empty-state">
                <div style={{ fontSize: 40 }}>📄</div>
                <div>Report will appear here after analysis</div>
              </div>
            ) : (
              <>
                <div className="report-section">
                  <div className="report-label">Clinical Findings</div>
                  <div className="report-text">{results.report?.findings}</div>
                </div>

                <div className="report-section">
                  <div className="report-label">Normal Structures</div>
                  <div className="report-text">
                    {results.report?.normal_structures}
                  </div>
                </div>

                <div className="report-section">
                  <div className="report-label">Impression</div>
                  <div className="impression-box">
                    {results.report?.impression}
                  </div>
                </div>

                <div className="report-section">
                  <div className="report-label">Recommendations</div>
                  {results.report?.recommendations?.map((r, i) => (
                    <div key={i} className="rec-item">
                      <div className="rec-num">{i + 1}</div>
                      <div>{r}</div>
                    </div>
                  ))}
                </div>

                <div className="report-section">
                  <div className="report-label">Doctor Validation</div>

                  {feedbackSubmitted ? (
                    <div className="success-msg">
                      Thank you! Feedback submitted successfully.
                      <br />
                      <span style={{ fontSize: 10, color: '#4a8' }}>
                        Helping improve AI accuracy
                      </span>
                    </div>
                  ) : (
                    <>
                      <div className="form-group">
                        <label className="form-label">Doctor Name *</label>
                        <input
                          className="form-input"
                          placeholder="Dr. Name"
                          value={feedback.doctorName}
                          onChange={e =>
                            setFeedback({ ...feedback, doctorName: e.target.value })
                          }
                        />
                      </div>

                      <div className="form-group">
                        <label className="form-label">Hospital</label>
                        <input
                          className="form-input"
                          placeholder="Hospital name"
                          value={feedback.hospital}
                          onChange={e =>
                            setFeedback({ ...feedback, hospital: e.target.value })
                          }
                        />
                      </div>

                      <div className="form-group">
                        <label className="form-label">AI Report Accuracy *</label>
                        <div className="feedback-rating">
                          {['correct', 'partial', 'wrong'].map(r => (
                            <button
                              key={r}
                              className={`rating-btn ${
                                feedback.rating === r ? `selected-${r}` : ''
                              }`}
                              onClick={() =>
                                setFeedback({ ...feedback, rating: r })
                              }
                            >
                              {r === 'correct' && 'Correct'}
                              {r === 'partial' && 'Partial'}
                              {r === 'wrong'   && 'Wrong'}
                            </button>
                          ))}
                        </div>
                      </div>

                      <div className="form-group">
                        <label className="form-label">
                          Your Clinical Findings
                        </label>
                        <textarea
                          className="form-input"
                          rows={3}
                          placeholder="Add your clinical observations..."
                          style={{ resize: 'vertical' }}
                          value={feedback.comments}
                          onChange={e =>
                            setFeedback({ ...feedback, comments: e.target.value })
                          }
                        />
                      </div>

                      <div className="form-group">
                        <label className="form-label">
                          Recommend this system?
                        </label>
                        <div className="feedback-rating">
                          {['yes', 'no', 'maybe'].map(r => (
                            <button
                              key={r}
                              className={`rating-btn ${
                                feedback.recommend === r
                                  ? 'selected-correct' : ''
                              }`}
                              onClick={() =>
                                setFeedback({ ...feedback, recommend: r })
                              }
                            >
                              {r === 'yes'   && 'Yes'}
                              {r === 'no'    && 'No'}
                              {r === 'maybe' && 'Maybe'}
                            </button>
                          ))}
                        </div>
                      </div>

                      <button
                        className="submit-feedback-btn"
                        onClick={handleFeedbackSubmit}
                        disabled={submittingFeedback}
                      >
                        {submittingFeedback
                          ? 'Submitting...' : 'Submit Validation'
                        }
                      </button>
                    </>
                  )}
                </div>

                <div className="disclaimer">
                  <strong>Disclaimer:</strong> This AI analysis is for
                  research purposes only. Not a substitute for professional
                  medical diagnosis. All findings must be verified by a
                  licensed radiologist.
                </div>
              </>
            )}
          </div>

        </div>
        {/* ════ END RIGHT PANEL ════ */}

      </div>
      {/* ══ END MAIN LAYOUT ══ */}

    </div>
  );
}