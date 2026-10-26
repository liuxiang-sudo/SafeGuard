document.addEventListener('DOMContentLoaded', function() {
    // Initialize time tracking
    const startTime = Date.now();
    const form = document.getElementById('detection-form');
    const timeInput = document.getElementById('time_to_fill');
    const typingPatternInput = document.getElementById('typing_pattern');
    const navigationInput = document.getElementById('navigation');
    
    // 平台选择功能增强
    const platformOptions = document.querySelectorAll('.platform-option');
    const platformInput = document.getElementById('platform');
    const platformFields = document.querySelectorAll('.platform-fields');
    
    if (platformOptions.length > 0) {
        platformOptions.forEach(option => {
            option.addEventListener('click', function() {
                // Add click effect
                this.classList.add('clicked');
                setTimeout(() => {
                    this.classList.remove('clicked');
                }, 300);
                
                // Remove active class from all platform options
                platformOptions.forEach(opt => opt.classList.remove('active'));
                
                // Add active class to the selected platform
                this.classList.add('active');
                
                // Set hidden form field value
                const platform = this.getAttribute('data-platform');
                platformInput.value = platform;
                
                // Hide all platform specific fields
                platformFields.forEach(field => field.classList.remove('active'));
                
                // Show platform specific fields
                const currentPlatformField = document.getElementById(`${platform}-fields`);
                if (currentPlatformField) {
                    currentPlatformField.classList.add('active');
                    
                    // Scroll to platform specific fields
                    setTimeout(() => {
                        currentPlatformField.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    }, 100);
                }
                
                // Add visual effects
                const img = this.querySelector('img');
                img.classList.add('bounce');
                setTimeout(() => {
                    img.classList.remove('bounce');
                }, 500);
            });
        });
    }
    
    // Track keystrokes and typing intervals
    let lastKeystrokeTime = 0;
    const keystrokeIntervals = [];
    
    // Track form field interactions
    const textInputs = document.querySelectorAll('input[type="text"], input[type="email"], textarea');
    
    textInputs.forEach(input => {
        input.addEventListener('keydown', function(event) {
            const now = Date.now();
            
            // Record the interval between keystrokes
            if (lastKeystrokeTime > 0) {
                const interval = now - lastKeystrokeTime;
                // Only record reasonable intervals (filter out extreme values)
                if (interval > 10 && interval < 5000) {
                    keystrokeIntervals.push(interval);
                }
            }
            
            lastKeystrokeTime = now;
        });
        
        // Add animation when getting focus
        input.addEventListener('focus', function() {
            this.parentNode.classList.add('active');
        });
        
        input.addEventListener('blur', function() {
            if (this.value === '') {
                this.parentNode.classList.remove('active');
            }
        });
    });
    
    // Set navigation count
    const navCount = sessionStorage.getItem('navCount') || 0;
    sessionStorage.setItem('navCount', parseInt(navCount) + 1);
    navigationInput.value = navCount;
    
    // Submit event
    if (form) {
        form.addEventListener('submit', function(event) {
            // Calculate time spent on form
            const timeSpent = (Date.now() - startTime) / 1000; // Convert to seconds
            timeInput.value = timeSpent.toFixed(2);
            
            // Record typing pattern
            typingPatternInput.value = keystrokeIntervals.join(',');
            
            // Set screen size and user agent
            document.getElementById('screen_size').value = `${window.innerWidth}x${window.innerHeight}`;
            document.getElementById('user_agent').value = navigator.userAgent;
            document.getElementById('referrer').value = document.referrer;
        });
    }
    
    // Animate elements on the results page
    const meterFills = document.querySelectorAll('.meter-fill');
    
    if (meterFills.length > 0) {
        // Add animation class after a short delay
        setTimeout(() => {
            meterFills.forEach(fill => {
                fill.style.width = fill.getAttribute('data-width') || fill.style.width;
            });
        }, 300);
    }
    
    // Add smooth scrolling to all internal links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth'
                });
            }
        });
    });
    
    // Click recording
    let clicks = [];
    document.addEventListener('click', function(e) {
        clicks.push(`${e.clientX},${e.clientY},${Date.now()}`);
        if (clicks.length > 20) clicks.shift();
        document.getElementById('click_pattern').value = clicks.join('|');
    });
    
    // Hide loading animation after page load completes
    window.addEventListener('load', function() {
        const loader = document.getElementById('loader');
        if (loader) {
            setTimeout(function() {
                loader.classList.add('hidden');
            }, 500);
        }
    });
    
    // 生成样本数据下载功能
    const generateBtn = document.getElementById('generate-sample-data');
    if (generateBtn) {
        generateBtn.addEventListener('click', function(e) {
            e.preventDefault();
            fetch('/download_template/general')
                .then(response => response.json())
                .then(data => {
                    const blob = new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'});
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = 'sample_general.json';
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    URL.revokeObjectURL(url);
                })
                .catch(() => {
                    alert('生成样本数据失败，请稍后重试');
                });
        });
    }
}); 