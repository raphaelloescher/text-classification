const app = Vue.createApp({
    data() {
        return {
            incident: '',
            recommendations: [],
            loading: false,
            confusionMatrixUrl: '/api/confusion-matrix',
            metrics: {
                'accuracy': 0.95,
                'f1': 0.93,
                'recall': 0.90,
                'precision': 0.94,
                'confusion_matrix': '[[45, 5], [3, 47]]'
            }
        };
    },
    methods: {
        async submitIncident() {
            if (!this.incident.trim()) {
                alert("Please describe the incident before submitting.");
                return;
            }
            
            this.loading = true;
            this.recommendations = [];

            try {
                const response = await fetch('/api/classify', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ description: this.incident })
                });

                if (!response.ok) {
                    throw new Error("Failed to fetch recommendations.");
                }

                const data = await response.json();
                this.recommendations = data.recommendations;
            } catch (error) {
                alert("Error fetching recommendations.");
                console.error(error);
            } finally {
                this.loading = false;
            }
        },
        async fetchMetrics() {
            const response = await fetch('/api/metrics');
            const data = await response.json();
            this.metrics = data;
        },
        // Open Modal for Metrics
        showMetrics() {
            this.confusionMatrixUrl = `/api/confusion-matrix?timestamp=${new Date().getTime()}`;
            const modal = new bootstrap.Modal(document.getElementById('metricsModal'));
            modal.show();
        },
        // Fill textarea with example descriptions
        fillExample(exampleText) {
            this.incident = exampleText;
        }
    }
});

app.mount('#app');