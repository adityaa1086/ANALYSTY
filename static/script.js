document.addEventListener('DOMContentLoaded', function() {
    let generateBtn = document.getElementById('generate-btn');
    let plotContainer = document.getElementById('plot');
    let industrySelect = document.getElementById('industry-select');
    let analysisResults = document.getElementById('analysis-results');
    let draggableElements = document.querySelectorAll('.file-rep');
    let dropZones = document.querySelectorAll('.drop-zone');

    draggableElements.forEach(elem => {
        elem.addEventListener('dragstart', function(event) {
            event.dataTransfer.setData('text/plain', event.target.id);
        });
    });

    dropZones.forEach(zone => {
        zone.addEventListener('dragover', function(event) {
            event.preventDefault(); // Necessary to allow dropping
            event.target.classList.add('drag-over');
        });

        zone.addEventListener('dragleave', function(event) {
            event.target.classList.remove('drag-over');
        });

        zone.addEventListener('drop', function(event) {
            event.preventDefault(); // Prevent default to allow drop
            event.target.classList.remove('drag-over');
            const data = event.dataTransfer.getData('text/plain');
            const draggableElement = document.getElementById(data);
            handleFileUpload(draggableElement.id, zone.id); // Pass the ID of the dragged element and the drop zone
        });
    });

    generateBtn.addEventListener('click', () => {
        let selectedIndustry = industrySelect.value;
        if (!selectedIndustry) {
            alert('Please select an industry.');
            return;
        }
        generateData(selectedIndustry);
    });

    function handleFileUpload(fileId, dropZoneId) {
        // Get the file object from the input element - this requires that the input element has the 'files' attribute
        let fileInput = document.getElementById(fileId);
        let file = fileInput.files[0];  // Get the first file from the file input
        if (!file) {
            console.error('No file selected for upload');
            return;
        }
    
        let formData = new FormData();
        formData.append('file', file);  // Append the actual file to the formData
    
        // Choose the correct endpoint based on the file and drop zone ID
        let uploadEndpoint = '';
        if (dropZoneId === 'drop-zone-1' && fileId === 'file1') {
            uploadEndpoint = '/process_buyers_customers';
        } else if (dropZoneId === 'drop-zone-2' && fileId === 'file2') {
            uploadEndpoint = '/process_sales_revenue';
        } else {
            console.error('File dropped in the wrong drop zone');
            return;
        }
    
        // Fetch API to POST the file to the server
        fetch(uploadEndpoint, {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            // Handle the response data here
            console.log('Success:', data);
            // Update the UI to show that the file has been processed
            let dropZone = document.getElementById(dropZoneId);
            dropZone.textContent = `File processed: ${fileId}`;
            dropZone.classList.add('drop-active'); // Add the gradient background
            if (data.shap_plot) {
                const shapPlotContainer = document.getElementById('shap-plot');
                Plotly.newPlot(shapPlotContainer, JSON.parse(data.shap_plot));
            }
        })
        .catch((error) => {
            // Handle errors here
            console.error('Error:', error);
            let dropZone = document.getElementById(dropZoneId);
            dropZone.textContent = 'File processing failed!';
        });
    }
    

    function generateData(industry) {
        fetch(`/generate?industry=${industry}`)
            .then(response => response.json())
            .then(data => {
                if (data.result) {
                    Plotly.newPlot(plotContainer, JSON.parse(data.result));
                }
                if (data.explanation) {
                    displayResults(data.explanation);
                }
            })
            .catch(error => {
                console.error('Error:', error);
            });
    }

    function displayResults(explanation) {
        analysisResults.innerHTML = `<p>${explanation}</p>`;
    }
});
