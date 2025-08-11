# 🎯 DAPP Dashboard Access Guide

## 🚀 Quick Start - Access Your Dashboard

### **Option 1: Simple HTML Dashboard (Ready Now!)**

1. **Start the Backend:**
   ```bash
   ./start_backend.sh
   # OR manually:
   source venv/bin/activate
   SECRET_KEY="development-secret-key-for-testing-32chars" ENCRYPTION_KEY="development-encryption-key-32chars" python main.py
   ```

2. **Open Simple Dashboard:**
   ```bash
   # Open the HTML dashboard in your browser
   open frontend/simple-dashboard.html
   # OR
   python -m http.server 8080 -d frontend
   # Then visit: http://localhost:8080/simple-dashboard.html
   ```

### **Option 2: Full Next.js Dashboard (After Setup)**

1. **Install Frontend Dependencies:**
   ```bash
   cd frontend
   npm install  # This may take a few minutes
   ```

2. **Start Next.js Development Server:**
   ```bash
   npm run dev
   # Visit: http://localhost:3000
   ```

## 🔧 Troubleshooting Dashboard Access

### **If you can't reach the web dashboard:**

1. **Check Backend Status:**
   ```bash
   curl http://localhost:8000/health
   ```
   Should return: `{"status":"healthy","version":"1.0.0","service":"Data Analysis Platform"}`

2. **Check Frontend Process:**
   ```bash
   # For Next.js dashboard
   curl http://localhost:3000
   
   # For simple HTML dashboard
   curl http://localhost:8080/simple-dashboard.html
   ```

3. **Common Issues & Solutions:**

   **❌ Backend not responding:**
   ```bash
   # Kill any existing processes
   pkill -f "python main.py"
   
   # Restart backend
   ./start_backend.sh
   ```

   **❌ Port already in use:**
   ```bash
   # Find process using port 8000
   lsof -i :8000
   # Kill the process
   kill -9 <PID>
   ```

   **❌ npm install hanging:**
   ```bash
   # Use yarn instead
   cd frontend
   yarn install
   yarn dev
   
   # OR try npm with different registry
   npm install --registry https://registry.npmjs.org/
   ```

## 📱 Available Interfaces

### **1. Simple HTML Dashboard**
- **URL:** `file:///.../frontend/simple-dashboard.html` or `http://localhost:8080/simple-dashboard.html`
- **Features:** Basic connection testing, health checks, quick links
- **Best for:** Quick status checks, troubleshooting
- **Requirements:** Just a web browser

### **2. Next.js Web Dashboard**
- **URL:** `http://localhost:3000`
- **Features:** Full interactive dashboard, real-time updates, mobile responsive
- **Best for:** Business users, stakeholders, production use
- **Requirements:** Node.js, npm/yarn

### **3. Jupyter Lab Interface**
- **URL:** `http://localhost:8888`
- **Features:** Interactive notebooks, widgets, deep analysis
- **Best for:** Data scientists, researchers, development
- **Requirements:** Python, Jupyter Lab
- **Start with:** `jupyter lab`

### **4. FastAPI Documentation**
- **URL:** `http://localhost:8000/docs`
- **Features:** Interactive API documentation, testing interface
- **Best for:** Developers, API integration
- **Requirements:** Running backend

## 🎯 Recommended Workflow

### **For Data Scientists:**
1. Start backend: `./start_backend.sh`
2. Open Jupyter: `jupyter lab`
3. Use hybrid template: `notebooks/templates/hybrid_workflow_template.ipynb`

### **For Business Users:**
1. Start backend: `./start_backend.sh`
2. Open web dashboard: Visit `http://localhost:3000` (after `npm run dev`)
3. Access from mobile/tablet for presentations

### **For Developers:**
1. Start backend: `./start_backend.sh`
2. Check API docs: `http://localhost:8000/docs`
3. Test with simple dashboard: `frontend/simple-dashboard.html`

## 🔍 Status Indicators

### **Backend Health:**
- ✅ **Healthy**: Returns JSON with status "healthy"
- ⚠️ **Starting**: Connection refused or timeout
- ❌ **Error**: Returns error message or 500 status

### **Frontend Status:**
- ✅ **Connected**: Can reach backend APIs
- ⚠️ **Loading**: Attempting connection
- ❌ **Disconnected**: Cannot reach backend

## 🚀 Performance Notes

### **Simple HTML Dashboard:**
- **Load time**: < 1 second
- **Memory usage**: ~10MB
- **Best for**: Quick checks, status monitoring

### **Next.js Dashboard:**
- **Load time**: 2-3 seconds (first visit), < 1 second (cached)
- **Memory usage**: ~50-100MB
- **Best for**: Production use, real-time operations

### **Jupyter Interface:**
- **Load time**: 3-5 seconds
- **Memory usage**: ~100-200MB
- **Best for**: Data analysis, model development

---

## 🆘 Need Help?

**If you still can't access the dashboard:**

1. **Check this guide:** All troubleshooting steps above
2. **Check logs:** Look at terminal output from `./start_backend.sh`
3. **Restart everything:**
   ```bash
   # Kill all processes
   pkill -f "python main.py"
   pkill -f "npm run dev"
   pkill -f "jupyter"
   
   # Restart backend
   ./start_backend.sh
   
   # In new terminal, restart frontend
   cd frontend && npm run dev
   ```

**The simple HTML dashboard should work immediately once the backend is running!** 🎉