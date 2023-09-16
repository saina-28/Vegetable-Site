const express = require('express');
const { getCurrentCartItems, getCheckoutData, removeCurrentCartItem, updateCartItem, removeCart } = require('../controllers/cartController');
const { checkCart } = require("../middlewares/checkCart");
const auth = require("../middlewares/auth");
const { addCheckoutANDCartData } = require('../middlewares/addCheckoutANDCartData')
const router = express.Router()


router.post('/get-current-cart-items', addCheckoutANDCartData, getCurrentCartItems)
router.post('/get-checkout-data', addCheckoutANDCartData, getCheckoutData)
router.post('/remove-current-cart-item', auth, removeCurrentCartItem)
router.post('/update-cart', auth, checkCart, updateCartItem)
router.post('/remove-cart', removeCart)


module.exports = router;