const express = require('express');
const { addItem, getItems, getAllItems, updateItem, removeItem, addInBulk } = require('../controllers/pantryController');
const router = express.Router()


router.post('/add', addItem)
router.post('/get', getItems)
router.post('/get-all', getAllItems)
router.post('/update-item', updateItem)
router.post('/remove', removeItem)


router.post('/add-items-bulk', addInBulk)

module.exports = router;