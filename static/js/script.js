// function runPythonScript() {
//     fetch('./pythonToPython/pythonToPython.py')
//         .then(response => response.text())
//         .then(data => console.log(data))
//         .catch(error => console.error('Error:', error));
// }

// document.getElementById('runScriptBtn').addEventListener('click', function() {
//     fetch('/run-script')
//         .then(response => {
//             if (!response.ok) {
//                 throw new Error('Network response was not ok');
//             }
//             return response.text();
//         })
//         .then(data => {
//             alert(data); // Display response from the server
//         })
//         .catch(error => {
//             console.error('There was a problem with the fetch operation:', error);
//         });
// });
function runPythonScript() {
    fetch('/run-script')
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.text();
        })
        .then(data => {
            alert(data); // Display response from the server
        })
        .catch(error => {
            console.error('There was a problem with the fetch operation:', error);
        });
}



