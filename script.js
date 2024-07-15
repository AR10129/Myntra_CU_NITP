document.addEventListener('DOMContentLoaded', (event) => {
    const slides = document.querySelectorAll('.slide');
    let index = 0;

    function showNextSlide() {
        slides[index].style.display = 'none';
        index = (index + 1) % slides.length;
        slides[index].style.display = 'block';
    }

    slides.forEach(slide => slide.style.display = 'none');
    slides[0].style.display = 'block';
    setInterval(showNextSlide, 4000); // Change slide every 4 seconds
});

document.addEventListener('DOMContentLoaded', function() {
    const skinColorPicker = document.getElementById('skinColorPicker');
    const hairColorPicker = document.getElementById('hairColorPicker');
    const eyeColorPicker = document.getElementById('eyeColorPicker');

    const skinColorBox = document.getElementById('skinColorBox');
    const hairColorBox = document.getElementById('hairColorBox');
    const eyeColorBox = document.getElementById('eyeColorBox');

    skinColorPicker.addEventListener('input', function() {
        const color = skinColorPicker.value;
        skinColorBox.style.backgroundColor = color;
    });

    hairColorPicker.addEventListener('input', function() {
        const color = hairColorPicker.value;
        hairColorBox.style.backgroundColor = color;
    });

    eyeColorPicker.addEventListener('input', function() {
        const color = eyeColorPicker.value;
        eyeColorBox.style.backgroundColor = color;
    });
});

document.getElementById('div2').addEventListener('submit', function(event) {
  event.preventDefault();

  const skinColor = document.getElementById('skinColorPicker').value;
  const hairColor = document.getElementById('hairColorPicker').value;
  const eyeColor = document.getElementById('eyeColorPicker').value;

  const data = {
    skinColor: skinColor,
    hairColor: hairColor,
    eyeColor: eyeColor
  };

