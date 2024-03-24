document.getElementById('file-upload').addEventListener('change', function(event) {
    const file = event.target.files[0];
    const reader = new FileReader();

    reader.onload = function(event) {
        const imageUrl = event.target.result;
        const imageContainer = document.getElementById('image-container');
        const imgElement = document.createElement('img');
        imgElement.src = imageUrl;
        imgElement.classList.add('uploaded-image');
        imageContainer.innerHTML = ''; // Clear previous image
        imageContainer.appendChild(imgElement);

    };

    reader.readAsDataURL(file);
});