const enBtn = document.getElementById('submitBtnEn');
const vnBtn = document.getElementById('submitBtnVn');

enBtn.addEventListener('click', function (e){
  callPredict("English");
})
vnBtn.addEventListener('click', function (e){
  callPredict("Vietnamese");
})

function callPredict(lang) {
  const fileInput = document.getElementById('file');
  const dewarpOpt = document.querySelector('#dewarp');
  const isDewarp = (dewarpOpt.checked) ? dewarpOpt.checked : false;
  const formData  = new FormData();
  formData.append('file', fileInput.files[0]);
  if (isDewarp) {
    formData.append('dewarp', true);
  } else {
    formData.append('dewarp', false);
  }
  // Language
  formData.append('lang', lang);

  showInfoBar();
  fetch('/predict/', {
    method: 'POST',
    body: formData,
  }).then(res => res.json()).then(data => {
    console.log(data);
    if (!data) {
      hideInfoBar();
      return
    }
    //document.getElementById('result').innerHTML=data['hip'];
    document.getElementById('prodpic').src = data['filename'];
    document.getElementById('prodpic').style.display = "inline-block";
    hideInfoBar();
  })
    .catch(error => {
      document.getElementById('result').innerHTML="";
      document.getElementById('prodpic').src = "";
      document.getElementById('prodpic').style.display = "none";
      hideInfoBar();
      console.error('Error:', error);
    });
}

const infobar = document.getElementById('infobar');
function showInfoBar() {
  infoBarTxtElem.innerHTML = "Uploading image ...";
  inst = setInterval(utb(), 5000);
  infobar.style.display = "block";
}

function hideInfoBar() {
  if (inst) {
    clearInterval(inst);
    inst = null;
  }
  infobar.style.display = "none";
}

var messages  = [
  "Calculating text points ...",
  "Converting points to spans ...",
  "Localizing text areas ...",
  "Preprocessing image ...",
  "Calculating opencv threshold ...",
  "Cropping text areas ...",
  "Running OCR ...",
  "Calculating total score ...",
  "Finding recommendations ...",
  "Results coming right up ...",
  "Holy crap this image is difficult ...",
  "This is taking longer than usual ...",
];
var infoBarTxtElem = document.getElementById("infobar_msg")
var inst = null;

function utb() {
  var counter = 0;
  return function() {
    infoBarTxtElem.innerHTML = messages[counter];
    counter++;
    if (counter >= messages.length) {
      counter = 0;
      if (inst) {
        clearInterval(inst);
      }
    }
  }
}
