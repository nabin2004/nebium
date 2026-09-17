window.HELP_IMPROVE_VIDEOJS = false;

var INTERP_BASE = "./static/interpolation/stacked";
var NUM_INTERP_FRAMES = 240;

var interp_images = [];
function preloadInterpolationImages() {
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_BASE + '/' + String(i).padStart(6, '0') + '.jpg';
    interp_images[i] = new Image();
    interp_images[i].src = path;
  }
}

function setInterpolationImage(i) {
  var image = interp_images[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper').empty().append(image);
}


$(document).ready(function() {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    var options = {
			slidesToScroll: 1,
			slidesToShow: 3,
			loop: true,
			infinite: true,
			autoplay: false,
			autoplaySpeed: 3000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
    	// Add listener to  event
    	carousels[i].on('before:show', state => {
    		console.log(state);
    	});
    }

    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
    	// bulmaCarousel instance is available as element.bulmaCarousel
    	element.bulmaCarousel.on('before-show', function(state) {
    		console.log(state);
    	});
    }

    /*var player = document.getElementById('interpolation-video');
    player.addEventListener('loadedmetadata', function() {
      $('#interpolation-slider').on('input', function(event) {
        console.log(this.value, player.duration);
        player.currentTime = player.duration / 100 * this.value;
      })
    }, false);*/
    if ($('#interpolation-slider').length) {
        preloadInterpolationImages();

        $('#interpolation-slider').on('input', function(event) {
          setInterpolationImage(this.value);
        });
        setInterpolationImage(0);
        $('#interpolation-slider').prop('max', NUM_INTERP_FRAMES - 1);
    }

    bulmaSlider.attach();

    // Interactive Demo handling
    $('.demo-btn').on('click', function() {
      $('.demo-btn').removeClass('is-primary').addClass('is-light');
      $(this).removeClass('is-light').addClass('is-primary');

      var promptKey = $(this).data('prompt');
      var demoData = {
        'e4e5': {
          prompt: 'e2e4 e7e5',
          continuations: 'g1f3 (King\'s Knight) b8c6 f1c4 (Italian Game) f8c5 c2c3 g8f6 d2d4 e5d4 c3d4 c5b4',
          legality: '100.0%',
          topMoves: [
            { move: 'g1f3', name: 'King\'s Knight Attack', prob: '84.2%' },
            { move: 'f1c4', name: 'Bishop\'s Opening', prob: '8.6%' },
            { move: 'b1c3', name: 'Vienna Game', prob: '4.7%' },
            { move: 'f2f4', name: 'King\'s Gambit', prob: '2.1%' }
          ]
        },
        'sicilian': {
          prompt: 'e2e4 c7c5',
          continuations: 'g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6 (Najdorf Defense) c1e3 e7e5 d4b3',
          legality: '100.0%',
          topMoves: [
            { move: 'g1f3', name: 'Open Sicilian Prep', prob: '78.5%' },
            { move: 'c2c3', name: 'Alapin Variation', prob: '11.3%' },
            { move: 'b1c3', name: 'Closed Sicilian', prob: '7.1%' },
            { move: 'd2d4', name: 'Smith-Morra / Halasz', prob: '2.4%' }
          ]
        },
        'd4d5': {
          prompt: 'd2d4 d7d5',
          continuations: 'c2c4 e7e6 b1c3 g8f6 c1g5 f8e7 e2e3 e8g8 g1f3 b8d7 a1c1 c7c6',
          legality: '100.0%',
          topMoves: [
            { move: 'c2c4', name: 'Queen\'s Gambit', prob: '81.9%' },
            { move: 'g1f3', name: 'Knight Variation', prob: '10.2%' },
            { move: 'c1f4', name: 'London System', prob: '5.8%' },
            { move: 'e2e3', name: 'Colle System', prob: '1.7%' }
          ]
        },
        'tactic': {
          prompt: 'e2e4 e7e5 g1f3 b8c6 f1c4 g8f6 f3g5 d7d5 e4d5 f6d5 g5f7 e8f7 d1f3 f7e6 b1c3',
          continuations: 'c6e7 d2d4 c7c6 c1g5 h7h6 g5e7 f8e7 d4e5',
          legality: '100.0%',
          topMoves: [
            { move: 'c6e7', name: 'Defense against Fork', prob: '72.4%' },
            { move: 'c6b4', name: 'Counter-threat', prob: '18.1%' },
            { move: 'e6e8', name: 'King Retreat (Blunder)', prob: '6.2%' },
            { move: 'd8d6', name: 'Queen Pin', prob: '2.8%' }
          ]
        }
      };

      var d = demoData[promptKey];
      if (d) {
        $('#demo-prompt').text(d.prompt);
        $('#demo-gen').text(d.continuations);
        $('#demo-legality').text(d.legality);
        
        var barsHtml = '';
        d.topMoves.forEach(function(m) {
          barsHtml += '<div style="margin-bottom: 12px;">' +
            '<div style="display: flex; justify-content: space-between; font-size: 0.9rem; margin-bottom: 4px;">' +
              '<span><strong>' + m.move + '</strong> <span class="has-text-grey">(' + m.name + ')</span></span>' +
              '<span><strong>' + m.prob + '</strong></span>' +
            '</div>' +
            '<progress class="progress is-info is-small" value="' + parseFloat(m.prob) + '" max="100">' + m.prob + '</progress>' +
          '</div>';
        });
        $('#demo-bars').html(barsHtml);
      }
    });

})

