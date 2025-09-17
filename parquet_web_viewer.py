#!/usr/bin/env python3
"""
Web-based Parquet data viewer
Usage:
    python parquet_web_viewer.py
    Then open http://localhost:8080 in your browser
"""

import os
import json
import base64
import ast
import codecs
import pandas as pd
from pathlib import Path
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Parquet Data Viewer</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        .controls { margin-bottom: 20px; }
        .controls select, .controls input, .controls button { 
            margin: 5px; padding: 8px; font-size: 14px; 
        }
        table { border-collapse: collapse; width: 100%; margin-top: 10px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .record { margin-bottom: 15px; padding: 12px; border: 1px solid #e0e0e0; border-radius: 8px; background: #fafafa; }
        .record-header { font-weight: bold; color: #2c3e50; margin-bottom: 8px; font-size: 16px; }
        .record-row { display: flex; flex-wrap: wrap; gap: 15px; align-items: flex-start; }
        .field { display: flex; align-items: flex-start; min-width: 200px; max-width: 400px; }
        .field-name { font-weight: bold; color: #34495e; min-width: 80px; margin-right: 8px; }
        .field-value { flex: 1; word-wrap: break-word; background: white; padding: 4px 8px; border-radius: 4px; border: 1px solid #ddd; }
        .long-text { max-height: 100px; overflow-y: auto; background: #f9f9f9; padding: 5px; }
        .image-container { margin: 10px 0; }
        .image-preview { max-width: 300px; max-height: 200px; border: 1px solid #ddd; border-radius: 5px; cursor: pointer; }
        .image-modal { display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; background-color: rgba(0,0,0,0.8); }
        .image-modal-content { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); max-width: 90%; max-height: 90%; }
        .image-modal img { max-width: 100%; max-height: 100%; }
        .close-modal { position: absolute; top: 10px; right: 25px; color: white; font-size: 35px; font-weight: bold; cursor: pointer; }
        .stats { background: #e8f4f8; padding: 10px; border-radius: 5px; margin-top: 20px; }
        .error { color: red; padding: 10px; background: #ffe6e6; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🗂️ Parquet Data Viewer</h1>
        <p>Browse and explore parquet files in the data directory</p>
    </div>
    
    <div class="controls">
        <select id="dirSelect" onchange="loadDirectory()">
            <option value="">Select a directory...</option>
        </select>
        <select id="fileSelect" onchange="loadFile()" disabled>
            <option value="">Select a parquet file...</option>
        </select>
        <input type="number" id="limitInput" placeholder="Limit (default: 10)" min="1" max="1000">
        <button onclick="loadFile()">Load Data</button>
        <button onclick="exportData()">Export Sample</button>
    </div>
    
    <div id="content"></div>
    
    <!-- Image Modal -->
    <div id="imageModal" class="image-modal" onclick="closeImageModal()">
        <span class="close-modal" onclick="closeImageModal()">&times;</span>
        <div class="image-modal-content">
            <img id="modalImage" src="" alt="Full size image">
        </div>
    </div>
    
    <script>
        // Load available directories on page load
        fetch('/api/directories')
            .then(response => response.json())
            .then(dirs => {
                const select = document.getElementById('dirSelect');
                dirs.forEach(dir => {
                    const option = document.createElement('option');
                    option.value = dir;
                    option.textContent = dir;
                    select.appendChild(option);
                });
            });
        
        function loadDirectory() {
            const dir = document.getElementById('dirSelect').value;
            const fileSelect = document.getElementById('fileSelect');
            
            // Clear file selection
            fileSelect.innerHTML = '<option value="">Select a parquet file...</option>';
            fileSelect.disabled = !dir;
            
            if (!dir) return;
            
            fetch(`/api/files?dir=${encodeURIComponent(dir)}`)
                .then(response => response.json())
                .then(files => {
                    files.forEach(file => {
                        const option = document.createElement('option');
                        option.value = file;
                        option.textContent = file.split('/').pop(); // Show only filename
                        fileSelect.appendChild(option);
                    });
                });
        }
        
        function loadFile() {
            const file = document.getElementById('fileSelect').value;
            const limit = document.getElementById('limitInput').value || 10;
            
            if (!file) {
                document.getElementById('content').innerHTML = '<div class="error">Please select a file</div>';
                return;
            }
            
            document.getElementById('content').innerHTML = '<p>Loading...</p>';
            
            fetch(`/api/data?file=${encodeURIComponent(file)}&limit=${limit}`)
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        document.getElementById('content').innerHTML = `<div class="error">${data.error}</div>`;
                        return;
                    }
                    displayData(data);
                })
                .catch(error => {
                    document.getElementById('content').innerHTML = `<div class="error">Error: ${error}</div>`;
                });
        }
        
        function displayData(data) {
            let html = `
                <div class="stats">
                    <h3>📊 File Statistics</h3>
                    <p><strong>File:</strong> ${data.file}</p>
                    <p><strong>Total Records:</strong> ${data.total_records}</p>
                    <p><strong>Columns:</strong> ${data.columns.length}</p>
                    <p><strong>Showing:</strong> ${data.records.length} records</p>
                </div>
                
                <h3>📋 Columns</h3>
                <table>
                    <tr><th>Name</th><th>Type</th></tr>
            `;
            
            data.columns.forEach(col => {
                html += `<tr><td>${col.name}</td><td>${col.type}</td></tr>`;
            });
            
            html += '</table><h3>🔍 Sample Data</h3>';
            
            data.records.forEach((record, idx) => {
                html += `
                    <div class="record">
                        <div class="record-header">Record ${idx + 1}</div>
                `;
                
                html += '<div class="record-row">';
                
                Object.entries(record).forEach(([key, value]) => {
                    let displayValue = value;
                    
                    // Check if this is an image field
                    if (key.toLowerCase().includes('image')) {
                        let imageData = null;
                        let imageType = 'jpeg';
                        let isValidImage = false;
                        let dataSource = 'unknown';
                        
                        // Handle different image data formats
                        if (typeof value === 'string' && value.length > 100) {
                            // Direct base64 string
                            imageData = value;
                            dataSource = 'direct_base64';
                            if (value.startsWith('/9j/') || value.startsWith('/9j')) {
                                imageType = 'jpeg';
                                isValidImage = true;
                            } else if (value.startsWith('iVBORw0KGgo')) {
                                imageType = 'png';
                                isValidImage = true;
                            } else if (value.match(/^[A-Za-z0-9+/=]+$/)) {
                                imageType = 'jpeg';
                                isValidImage = true;
                            }
                        } else if (typeof value === 'object' && value !== null) {
                            // Handle converted binary data
                            if (value.type === 'binary_converted' && value.data) {
                                imageData = value.data;
                                dataSource = 'converted_binary';
                                imageType = value.mime ? value.mime : 'jpeg'; // use server-provided mime if available
                                isValidImage = true;
                            }
                            // Complex object - try to extract base64 data
                            else if (value.bytes && typeof value.bytes === 'string') {
                                imageData = value.bytes;
                                dataSource = 'object_bytes';
                                isValidImage = true;
                            } else if (value.data && typeof value.data === 'string') {
                                imageData = value.data;
                                dataSource = 'object_data';
                                isValidImage = true;
                            } else if (typeof value === 'object') {
                                // Try to find base64-like strings in the object
                                const objStr = JSON.stringify(value);
                                const base64Match = objStr.match(/"([A-Za-z0-9+/=]{100,})"/);
                                if (base64Match) {
                                    imageData = base64Match[1];
                                    dataSource = 'extracted_from_json';
                                    isValidImage = true;
                                }
                            }
                        }
                        
                        if (isValidImage && imageData) {
                            const imageId = `img_${idx}_${key}`;
                            displayValue = `
                                <div class="image-container">
                                    <div style="margin-bottom: 5px; font-size: 12px; color: #666;">
                                        Image data (${Math.round(imageData.length * 0.75 / 1024)}KB, ${imageType.toUpperCase()})
                                    </div>
                                    <img id="${imageId}" class="image-preview" 
                                         src="data:image/${imageType};base64,${imageData}" 
                                         alt="Image preview" 
                                         onclick="showImageModal('${imageId}')"
                                         onload="console.log('Image loaded successfully')"
                                         onerror="console.log('Image failed to load'); this.style.display='none'; this.nextElementSibling.style.display='block';">
                                    <div style="display:none; color: #999; font-style: italic; padding: 10px; border: 1px dashed #ccc;">
                                        ❌ Failed to display image<br>
                                        <small>Data structure: ${typeof value}</small>
                                    </div>
                                </div>
                            `;
                        } else {
                            // Show the complex object structure
                            if (typeof value === 'object') {
                                displayValue = `
                                    <div style="margin-bottom: 10px;">
                                        <div style="font-size: 12px; color: #666; margin-bottom: 5px;">
                                            Complex image object (${typeof value})
                                        </div>
                                        <pre style="background: #f5f5f5; padding: 10px; border-radius: 3px; max-height: 200px; overflow-y: auto;">${JSON.stringify(value, null, 2)}</pre>
                                    </div>
                                `;
                            } else {
                                displayValue = `<div class="long-text">${value}</div>`;
                            }
                        }
                    } else if (typeof value === 'string' && value.length > 200) {
                        displayValue = `<div class="long-text">${value}</div>`;
                    } else if (typeof value === 'object') {
                        displayValue = `<pre>${JSON.stringify(value, null, 2)}</pre>`;
                    }
                    
                    html += `
                        <div class="field">
                            <span class="field-name">${key}:</span>
                            <div class="field-value">${displayValue}</div>
                        </div>
                    `;
                });
                
                html += '</div></div>'; // Close record-row and record
            });
            
            document.getElementById('content').innerHTML = html;
        }
        
        function showImageModal(imageId) {
            const img = document.getElementById(imageId);
            const modal = document.getElementById('imageModal');
            const modalImg = document.getElementById('modalImage');
            
            modalImg.src = img.src;
            modal.style.display = 'block';
        }
        
        function closeImageModal() {
            document.getElementById('imageModal').style.display = 'none';
        }
        
        function exportData() {
            const file = document.getElementById('fileSelect').value;
            const limit = document.getElementById('limitInput').value || 10;
            
            if (!file) {
                alert('Please select a file first');
                return;
            }
            
            window.open(`/api/export?file=${encodeURIComponent(file)}&limit=${limit}`);
        }
    </script>
</body>
</html>
"""

# Helper utilities for image extraction/normalization
def _guess_image_mime_from_bytes(b: bytes) -> str:
    """Best-effort detection of common image types from header bytes."""
    try:
        if b.startswith(b"\xff\xd8\xff"):
            return "jpeg"
        if b.startswith(b"\x89PNG\r\n\x1a\n"):
            return "png"
        if b[:6] in (b"GIF87a", b"GIF89a"):
            return "gif"
        if b.startswith(b"RIFF") and b[8:12] == b"WEBP":
            return "webp"
    except Exception:
        pass
    return "jpeg"


def _is_base64_like(s: str) -> bool:
    if not isinstance(s, str) or len(s) < 32:
        return False
    # Heuristic: only base64 characters and length multiple of 4
    import re
    if not re.fullmatch(r"[A-Za-z0-9+/=\n\r]+", s):
        return False
    return (len(s.strip()) % 4) == 0


def _try_convert_image_value_to_b64(value):
    """Try to convert various image representations to a base64 string.
    Returns (base64_str, mime, source) or (None, None, None).
    """
    # Direct bytes-like
    if isinstance(value, (bytes, bytearray, memoryview)):
        b = bytes(value)
        return base64.b64encode(b).decode("utf-8"), _guess_image_mime_from_bytes(b), "bytes"

    # Dict-like containers that may hold bytes/base64
    if isinstance(value, dict):
        for k in ("bytes", "data", "value", "content"):
            v = value.get(k)
            if isinstance(v, (bytes, bytearray, memoryview)):
                b = bytes(v)
                return base64.b64encode(b).decode("utf-8"), _guess_image_mime_from_bytes(b), f"dict:{k}_bytes"
            if isinstance(v, str):
                # Try to parse string below
                maybe, mime, source = _try_convert_image_value_to_b64(v)
                if maybe:
                    return maybe, mime, f"dict:{k}_str->{source}"

    # Strings: could be base64, a bytes-literal, or escape-sequence string
    if isinstance(value, str):
        s = value.strip()

        # 1) base64 directly
        if _is_base64_like(s) and len(s) > 128:
            # Best effort; leave mime as jpeg by default
            return s, "jpeg", "direct_base64"

        # 2) Python bytes literal: b'...' or b"..."
        if (s.startswith("b'") and s.endswith("'")) or (s.startswith('b"') and s.endswith('"')):
            try:
                obj = ast.literal_eval(s)
                if isinstance(obj, (bytes, bytearray)):
                    b = bytes(obj)
                    return base64.b64encode(b).decode("utf-8"), _guess_image_mime_from_bytes(b), "bytes_literal"
            except Exception:
                pass

        # 3) Escape sequence heavy strings e.g. "\xff\xd8...JFIF..."
        if "\\x" in s or "JFIF" in s or s.startswith("\\xff\\xd8"):
            try:
                # Decode escapes to text then encode to bytes 1:1
                unescaped = codecs.decode(s, "unicode_escape")
                b = unescaped.encode("latin1", errors="ignore")
                if len(b) > 32:
                    return base64.b64encode(b).decode("utf-8"), _guess_image_mime_from_bytes(b), "escape_string"
            except Exception:
                pass

    return None, None, None

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/directories')
def get_directories():
    """Get list of directories under data folder recursively"""
    data_dir = Path('data')
    if not data_dir.exists():
        return jsonify([])
    
    directories = ['data']  # Include root data directory
    
    def scan_directories(path):
        for item in path.iterdir():
            if item.is_dir():
                rel_path = str(item.relative_to(Path('.')))
                directories.append(rel_path)
                scan_directories(item)
    
    scan_directories(data_dir)
    return jsonify(sorted(directories))

@app.route('/api/files')
def get_files():
    """Get list of parquet files in specified directory"""
    dir_path = request.args.get('dir', 'data')
    target_dir = Path(dir_path)
    
    if not target_dir.exists():
        return jsonify([])
    
    parquet_files = []
    for file in target_dir.glob('*.parquet'):
        parquet_files.append(str(file))
    
    return jsonify(sorted(parquet_files))

@app.route('/api/data')
def get_data():
    """Get parquet file data"""
    file_path = request.args.get('file')
    limit = int(request.args.get('limit', 10))
    
    if not file_path:
        return jsonify({'error': 'No file specified'})
    
    try:
        df = pd.read_parquet(file_path)
        
        # Get column info
        columns = []
        for col in df.columns:
            columns.append({
                'name': col,
                'type': str(df[col].dtype)
            })
        
        # Get sample records
        sample_df = df.head(limit)
        records = []
        for _, row in sample_df.iterrows():
            record = {}
            for col in df.columns:
                value = row[col]
                col_l = str(col).lower()
                if pd.isna(value):
                    value = None
                    record[col] = value
                    continue

                # If value is bytes-like or column name suggests image, try to convert
                is_image_like_col = any(k in col_l for k in ["image", "img", "picture", "photo", "frames"]) or isinstance(value, (bytes, bytearray, memoryview))
                if is_image_like_col:
                    b64, mime, source = _try_convert_image_value_to_b64(value)
                    if b64:
                        record[col] = {
                            'type': 'binary_converted',
                            'data': b64,
                            'mime': mime,
                            'source': source
                        }
                        continue

                # Preserve complex objects for frontend inspection
                if isinstance(value, (dict, list)):
                    record[col] = value
                # Strings: keep as-is (frontend will show long-text)
                elif isinstance(value, str):
                    record[col] = value
                # Bytes-like but not recognized as image: still convert to base64 raw
                elif isinstance(value, (bytes, bytearray, memoryview)):
                    b = bytes(value)
                    record[col] = {
                        'type': 'binary_converted',
                        'data': base64.b64encode(b).decode('utf-8'),
                        'mime': _guess_image_mime_from_bytes(b),
                        'source': 'bytes_fallback'
                    }
                else:
                    record[col] = str(value)
            records.append(record)
        
        return jsonify({
            'file': file_path,
            'total_records': len(df),
            'columns': columns,
            'records': records
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/export')
def export_data():
    """Export sample data as JSON"""
    file_path = request.args.get('file')
    limit = int(request.args.get('limit', 10))
    
    if not file_path:
        return jsonify({'error': 'No file specified'})
    
    try:
        df = pd.read_parquet(file_path)
        sample_df = df.head(limit)
        
        # Convert to JSON
        data = sample_df.to_dict('records')
        
        return jsonify(data)
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print("🚀 Starting Parquet Web Viewer...")
    print("📁 Looking for parquet files in 'data' directory")
    print("🌐 Open http://localhost:10021 in your browser")
    print("⏹️  Press Ctrl+C to stop")
    
    app.run(host='0.0.0.0', port=10021, debug=True)
