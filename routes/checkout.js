const express = require('express');
const router = express.Router()
const { createCheckoutSession, cancelEvent, stripeWebHookResponse } = require('../controllers/checkoutController')
const { addCheckoutANDCartData } = require('../middlewares/addCheckoutANDCartData')
const auth  = require('../middlewares/auth')
const { validateOrder } = require('../middlewares/validateOrder')
const { pushtoTransit } = require('../middlewares/pushtoTransit')



const addUserStripeID = require('../middlewares/addUserStripeID')
router.post('/process-order', auth, addCheckoutANDCartData, addUserStripeID, validateOrder, pushtoTransit, createCheckoutSession)
router.post('/webhook', express.raw({type: 'application/json'}), stripeWebHookResponse);


router.get('/cancel', cancelEvent)
module.exports = router 