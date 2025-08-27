// LangGraph Research Workflow UI JavaScript

class ResearchWorkflowUI {
    constructor() {
        this.apiBase = window.location.origin;
        this.currentRequest = null;
        this.init();
    }

    init() {
        this.bindEvents();
        this.checkClusterStatus();
        
        // Check cluster status every 30 seconds
        setInterval(() => this.checkClusterStatus(), 30000);
    }

    bindEvents() {
        // Form submission
        const form = document.getElementById('research-form');
        form.addEventListener('submit', (e) => this.handleSubmit(e));

        // Example query buttons
        const exampleButtons = document.querySelectorAll('.example-query');
        exampleButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                const query = e.target.dataset.query;
                document.getElementById('query').value = query;
            });
        });
    }

    async handleSubmit(e) {
        e.preventDefault();
        
        const query = document.getElementById('query').value.trim();
        if (!query) {
            this.showError('Please enter a research query');
            return;
        }

        await this.executeResearch(query);
    }

    async executeResearch(query) {
        try {
            // Show loading state
            this.showLoading();
            this.updateProgress('Starting workflow...');
            
            // Disable form
            const submitBtn = document.getElementById('submit-btn');
            const queryInput = document.getElementById('query');
            submitBtn.disabled = true;
            submitBtn.textContent = '🔄 Researching...';
            queryInput.disabled = true;

            // Make API request
            const response = await fetch(`${this.apiBase}/api/research`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: query })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();

            if (result.success) {
                this.showResults(result);
            } else {
                this.showError(result.error || 'Research failed');
            }

        } catch (error) {
            console.error('Research failed:', error);
            this.showError(`Research failed: ${error.message}`);
        } finally {
            // Re-enable form
            const submitBtn = document.getElementById('submit-btn');
            const queryInput = document.getElementById('query');
            submitBtn.disabled = false;
            submitBtn.textContent = '🚀 Start Research';
            queryInput.disabled = false;
            
            this.hideProgress();
        }
    }

    showLoading() {
        document.getElementById('welcome-message').classList.add('hidden');
        document.getElementById('results-content').classList.add('hidden');
        document.getElementById('error-message').classList.add('hidden');
        document.getElementById('loading-message').classList.remove('hidden');
        document.getElementById('progress-panel').classList.remove('hidden');
        
        // Simulate progress steps
        this.simulateProgress();
    }

    simulateProgress() {
        const steps = ['planning', 'search', 'analysis', 'finalize'];
        const stepTexts = {
            'planning': 'Planning research strategy...',
            'search': 'Searching for information...',
            'analysis': 'Analyzing findings...',
            'finalize': 'Finalizing results...'
        };

        let currentStep = 0;
        
        const progressInterval = setInterval(() => {
            if (currentStep < steps.length) {
                // Mark current step as active
                this.setStepStatus(steps[currentStep], 'active');
                this.updateProgress(stepTexts[steps[currentStep]]);
                
                // Mark previous step as completed
                if (currentStep > 0) {
                    this.setStepStatus(steps[currentStep - 1], 'completed');
                }
                
                currentStep++;
            } else {
                clearInterval(progressInterval);
                // Mark final step as completed
                this.setStepStatus(steps[steps.length - 1], 'completed');
                this.updateProgress('Research complete!');
            }
        }, 2000); // Update every 2 seconds

        // Store interval to clear it if needed
        this.progressInterval = progressInterval;
    }

    setStepStatus(stepName, status) {
        const stepElement = document.querySelector(`[data-step="${stepName}"]`);
        if (stepElement) {
            stepElement.classList.remove('active', 'completed');
            if (status) {
                stepElement.classList.add(status);
            }
        }
    }

    updateProgress(text) {
        const currentStepElement = document.getElementById('current-step');
        if (currentStepElement) {
            currentStepElement.textContent = text;
        }
    }

    hideProgress() {
        document.getElementById('progress-panel').classList.add('hidden');
        
        // Clear progress interval
        if (this.progressInterval) {
            clearInterval(this.progressInterval);
            this.progressInterval = null;
        }
        
        // Reset progress steps
        const steps = document.querySelectorAll('.progress-step');
        steps.forEach(step => {
            step.classList.remove('active', 'completed');
        });
    }

    showResults(result) {
        document.getElementById('loading-message').classList.add('hidden');
        document.getElementById('error-message').classList.add('hidden');
        document.getElementById('welcome-message').classList.add('hidden');
        
        const resultsContent = document.getElementById('results-content');
        resultsContent.classList.remove('hidden');

        // Use formatted HTML if available, otherwise convert markdown
        let htmlContent = result.formatted_html;
        if (!htmlContent && result.result.final_result) {
            htmlContent = marked.parse(result.result.final_result);
        }

        if (htmlContent) {
            resultsContent.innerHTML = htmlContent;
        } else {
            resultsContent.innerHTML = '<p class="text-gray-600">No results generated.</p>';
        }

        // Highlight code blocks if Prism is available
        if (window.Prism) {
            window.Prism.highlightAllUnder(resultsContent);
        }

        // Scroll to results
        resultsContent.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    showError(message) {
        document.getElementById('loading-message').classList.add('hidden');
        document.getElementById('results-content').classList.add('hidden');
        document.getElementById('welcome-message').classList.add('hidden');
        
        const errorMessage = document.getElementById('error-message');
        errorMessage.classList.remove('hidden');
        
        document.getElementById('error-details').textContent = message;
    }

    async checkClusterStatus() {
        try {
            const response = await fetch(`${this.apiBase}/api/cluster/status`);
            const status = await response.json();
            
            this.updateClusterStatusUI(status);
        } catch (error) {
            console.error('Failed to check cluster status:', error);
            this.updateClusterStatusUI({ overall: 'unhealthy', error: error.message });
        }
    }

    updateClusterStatusUI(status) {
        const statusElement = document.getElementById('cluster-status');
        const indicator = statusElement.querySelector('div');
        const text = statusElement.querySelector('span');

        // Remove existing status classes
        indicator.classList.remove('bg-green-500', 'bg-yellow-500', 'bg-red-500', 'bg-gray-400');

        switch (status.overall) {
            case 'healthy':
                indicator.classList.add('bg-green-500');
                text.textContent = 'Cluster Healthy';
                break;
            case 'degraded':
                indicator.classList.add('bg-yellow-500');
                text.textContent = 'Cluster Degraded';
                break;
            case 'unhealthy':
                indicator.classList.add('bg-red-500');
                text.textContent = 'Cluster Unhealthy';
                break;
            default:
                indicator.classList.add('bg-gray-400');
                text.textContent = 'Status Unknown';
        }

        // Add tooltip with service details if available
        if (status.services) {
            const serviceCount = Object.keys(status.services).length;
            const healthyCount = Object.values(status.services).filter(s => s.status === 'healthy').length;
            text.title = `${healthyCount}/${serviceCount} services healthy`;
        }
    }
}

// Initialize the UI when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new ResearchWorkflowUI();
});
