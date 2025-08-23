/* Project Mango Documentation JavaScript */

document.addEventListener('DOMContentLoaded', function() {
    
    // Add smooth scrolling to anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Add copy functionality to code blocks
    addCopyButtons();
    
    // Initialize progress tracking
    initializeProgressTracking();
    
    // Add interactive elements
    addInteractiveFeatures();
    
    // Initialize search enhancements
    enhanceSearch();
    
    // Add keyboard shortcuts
    addKeyboardShortcuts();
});

function addCopyButtons() {
    // Add copy buttons to code blocks that don't already have them
    document.querySelectorAll('pre code').forEach((block, index) => {
        const pre = block.parentElement;
        if (!pre.querySelector('.copy-button')) {
            const button = document.createElement('button');
            button.className = 'copy-button';
            button.innerHTML = '📋';
            button.title = 'Copy code';
            button.addEventListener('click', () => copyCode(block, button));
            pre.style.position = 'relative';
            pre.appendChild(button);
        }
    });
}

function copyCode(block, button) {
    const text = block.textContent;
    navigator.clipboard.writeText(text).then(() => {
        button.innerHTML = '✅';
        button.title = 'Copied!';
        setTimeout(() => {
            button.innerHTML = '📋';
            button.title = 'Copy code';
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy text: ', err);
        button.innerHTML = '❌';
        setTimeout(() => {
            button.innerHTML = '📋';
        }, 2000);
    });
}

function initializeProgressTracking() {
    // Track user progress through documentation
    const currentPage = window.location.pathname;
    const visitedPages = JSON.parse(localStorage.getItem('mol_visited_pages') || '[]');
    
    if (!visitedPages.includes(currentPage)) {
        visitedPages.push(currentPage);
        localStorage.setItem('mol_visited_pages', JSON.stringify(visitedPages));
    }
    
    // Update progress indicators
    updateProgressIndicators(visitedPages);
}

function updateProgressIndicators(visitedPages) {
    // Calculate and display progress for different sections
    const sections = {
        'tutorials': {
            total: 9,
            pattern: /\/tutorials\//
        },
        'examples': {
            total: 4,
            pattern: /\/examples\//
        },
        'api': {
            total: 15,
            pattern: /\/api\//
        }
    };
    
    Object.entries(sections).forEach(([section, config]) => {
        const visited = visitedPages.filter(page => config.pattern.test(page)).length;
        const progress = Math.min((visited / config.total) * 100, 100);
        
        const progressBar = document.querySelector(`#${section}-progress`);
        if (progressBar) {
            progressBar.style.width = `${progress}%`;
            progressBar.setAttribute('aria-valuenow', progress);
        }
    });
}

function addInteractiveFeatures() {
    // Add interactive code examples
    addCodePlaygrounds();
    
    // Add collapsible sections
    addCollapsibleSections();
    
    // Add tooltips
    addTooltips();
    
    // Add difficulty indicators
    addDifficultyIndicators();
}

function addCodePlaygrounds() {
    // Find code blocks marked as interactive
    document.querySelectorAll('.language-python[data-interactive="true"]').forEach(block => {
        const container = document.createElement('div');
        container.className = 'code-playground';
        
        const runButton = document.createElement('button');
        runButton.className = 'run-button';
        runButton.innerHTML = '▶️ Run Code';
        runButton.addEventListener('click', () => runCode(block));
        
        const output = document.createElement('div');
        output.className = 'code-output';
        output.innerHTML = '<em>Click "Run Code" to execute</em>';
        
        block.parentElement.appendChild(container);
        container.appendChild(runButton);
        container.appendChild(output);
    });
}

function runCode(block) {
    // Simulate code execution (in real implementation, this would use Pyodide or similar)
    const output = block.parentElement.querySelector('.code-output');
    output.innerHTML = '<em>🔄 Running code...</em>';
    
    setTimeout(() => {
        output.innerHTML = `
            <div class="mock-output">
                <span class="output-success">✅ Code executed successfully!</span><br>
                <span class="output-result">Example output would appear here.</span><br>
                <small><em>Note: This is a mock execution. In production, this would run actual Python code.</em></small>
            </div>
        `;
    }, 1500);
}

function addCollapsibleSections() {
    // Make certain sections collapsible
    document.querySelectorAll('.collapsible').forEach(section => {
        const header = section.querySelector('h3, h4');
        if (header) {
            header.style.cursor = 'pointer';
            header.addEventListener('click', () => {
                section.classList.toggle('collapsed');
            });
        }
    });
}

function addTooltips() {
    // Add tooltips to technical terms
    const tooltips = {
        'MoL': 'Modular Layer - A system for dynamic fusion of transformer layers',
        'SLERP': 'Spherical Linear Interpolation - A method for smoothly interpolating between model weights',
        'TIES': 'Trim, Elect Sign, Disjoint Merge - An advanced model merging technique',
        'Adapter': 'A component that handles dimensional mismatches between models',
        'Router': 'A component that decides which expert model to use for each input'
    };
    
    Object.entries(tooltips).forEach(([term, description]) => {
        const regex = new RegExp(`\\b${term}\\b`, 'g');
        document.querySelectorAll('p, li').forEach(element => {
            if (!element.querySelector('code, pre')) {
                element.innerHTML = element.innerHTML.replace(regex, 
                    `<span class="tooltip" data-tooltip="${description}">${term}</span>`
                );
            }
        });
    });
}

function addDifficultyIndicators() {
    // Add difficulty stars to tutorials
    const difficulties = {
        'basic-fusion.md': 1,
        'adapters.md': 2,
        'routers.md': 2,
        'model-merging.md': 2,
        'training.md': 3,
        'large-models.md': 3,
        'multi-architecture.md': 3,
        'custom-components.md': 4,
        'deployment.md': 4
    };
    
    const currentPage = window.location.pathname.split('/').pop();
    const difficulty = difficulties[currentPage];
    
    if (difficulty) {
        const difficultyContainer = document.createElement('div');
        difficultyContainer.className = 'difficulty-indicator';
        difficultyContainer.innerHTML = `
            <span class="difficulty-label">Difficulty: </span>
            ${Array.from({length: 5}, (_, i) => 
                `<span class="difficulty-star ${i < difficulty ? 'filled' : ''}">⭐</span>`
            ).join('')}
        `;
        
        const firstH1 = document.querySelector('h1');
        if (firstH1) {
            firstH1.after(difficultyContainer);
        }
    }
}

function enhanceSearch() {
    // Add search suggestions and highlighting
    const searchInput = document.querySelector('.md-search__input');
    if (searchInput) {
        searchInput.addEventListener('input', function() {
            const query = this.value.toLowerCase();
            if (query.length > 2) {
                highlightSearchTerms(query);
            }
        });
    }
}

function highlightSearchTerms(query) {
    // Highlight search terms in the current page
    const walker = document.createTreeWalker(
        document.querySelector('.md-content'),
        NodeFilter.SHOW_TEXT,
        null,
        false
    );
    
    const textNodes = [];
    let node;
    while (node = walker.nextNode()) {
        textNodes.push(node);
    }
    
    textNodes.forEach(textNode => {
        if (textNode.textContent.toLowerCase().includes(query)) {
            const parent = textNode.parentNode;
            if (parent && !parent.classList.contains('highlight')) {
                const highlighted = textNode.textContent.replace(
                    new RegExp(query, 'gi'),
                    `<mark class="search-highlight">$&</mark>`
                );
                const span = document.createElement('span');
                span.innerHTML = highlighted;
                parent.replaceChild(span, textNode);
            }
        }
    });
}

function addKeyboardShortcuts() {
    // Add useful keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + K for search
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            const searchInput = document.querySelector('.md-search__input');
            if (searchInput) {
                searchInput.focus();
            }
        }
        
        // Escape to close search
        if (e.key === 'Escape') {
            const searchInput = document.querySelector('.md-search__input');
            if (searchInput && document.activeElement === searchInput) {
                searchInput.blur();
            }
        }
        
        // Arrow keys for navigation
        if (e.key === 'ArrowLeft' && e.altKey) {
            const prevLink = document.querySelector('.md-footer__link--prev');
            if (prevLink) prevLink.click();
        }
        
        if (e.key === 'ArrowRight' && e.altKey) {
            const nextLink = document.querySelector('.md-footer__link--next');
            if (nextLink) nextLink.click();
        }
    });
}

// Analytics and feedback
function trackUserInteraction(action, element) {
    // Track user interactions for analytics
    if (typeof gtag !== 'undefined') {
        gtag('event', action, {
            'custom_parameter': element
        });
    }
}

// Add event listeners for tracking
document.addEventListener('click', function(e) {
    if (e.target.matches('.copy-button')) {
        trackUserInteraction('copy_code', e.target.previousElementSibling.className);
    }
    
    if (e.target.matches('.run-button')) {
        trackUserInteraction('run_code', 'interactive_example');
    }
    
    if (e.target.matches('a[href^="http"]')) {
        trackUserInteraction('external_link', e.target.href);
    }
});

// Feedback widget
function addFeedbackWidget() {
    const widget = document.createElement('div');
    widget.className = 'feedback-widget';
    widget.innerHTML = `
        <div class="feedback-question">Was this page helpful?</div>
        <div class="feedback-buttons">
            <button class="feedback-yes" onclick="submitFeedback(true)">👍 Yes</button>
            <button class="feedback-no" onclick="submitFeedback(false)">👎 No</button>
        </div>
        <div class="feedback-thanks" style="display: none;">
            Thank you for your feedback! 🙏
        </div>
    `;
    
    const content = document.querySelector('.md-content');
    if (content) {
        content.appendChild(widget);
    }
}

function submitFeedback(positive) {
    // Submit feedback (implement with your analytics system)
    trackUserInteraction('feedback', positive ? 'positive' : 'negative');
    
    const widget = document.querySelector('.feedback-widget');
    widget.querySelector('.feedback-question').style.display = 'none';
    widget.querySelector('.feedback-buttons').style.display = 'none';
    widget.querySelector('.feedback-thanks').style.display = 'block';
    
    setTimeout(() => {
        widget.style.opacity = '0';
        setTimeout(() => widget.remove(), 300);
    }, 2000);
}

// Initialize feedback widget after page load
window.addEventListener('load', function() {
    setTimeout(addFeedbackWidget, 2000);
});

// Performance monitoring
function monitorPerformance() {
    // Monitor page load performance
    window.addEventListener('load', function() {
        const perfData = performance.getEntriesByType('navigation')[0];
        const loadTime = perfData.loadEventEnd - perfData.loadEventStart;
        
        if (loadTime > 3000) {
            console.warn('Page load time is slow:', loadTime + 'ms');
        }
        
        trackUserInteraction('page_load_time', loadTime);
    });
}

monitorPerformance();